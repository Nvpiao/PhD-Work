import torch
import torch.nn as nn

import numpy as np
import os
import random

import sys
sys.path.append('./src')


from src import utils, model

parser = utils.argument_parser()
args = parser.parse_args()
if args.ELMo:
    print("**ELMo word Embeddings!")
    parser.set_defaults(learning_rate_step=2,
                        embed_dim=256, GoogleEmbedding=False)
else:
    print("**word2vec Embeddings!")
args = parser.parse_args()

# Device Setting
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    print(f"Running on GPU")
    torch.cuda.manual_seed(2022)
else:
    args.device = torch.device('cpu')
    print("Running on CPU")
    torch.manual_seed(2022)


random.seed(0)
torch.manual_seed(6)


utils.print_args(args)

# vocabs contain all vocab + <pad>, <bos>, <eos>, <unk>
args.vocabs = utils.load_file(args.vocab_path, file_type='json')
args.vocabs += ['<bos>', '<eos>']
args.n_vocabs = len(args.vocabs)
args.word2idx = {tok: i for i, tok in enumerate(args.vocabs)}
args.idx2word = {i: tok for i, tok in enumerate(args.vocabs)}
args.padding_idx = args.word2idx[args.padding_symbol]

# Sentence encoder
sentence_encoder = model.SentenceEmbeddingModel(args).to(args.device)
# Convolution layer for extracting global coherence patterns
global_feature_extractor = model.LightweightConvolution(args).to(args.device)
# Bilinear layer for modeling inter-sentence relation
bilinear_layer = model.BiAffine(args).to(args.device)
# Linear layer
coherence_scorer = model.LocalCoherenceScore(args).to(args.device)
local_global_model = nn.Sequential(sentence_encoder,
                                   bilinear_layer,
                                   global_feature_extractor,
                                   coherence_scorer)
optimizer = torch.optim.Adam(
    local_global_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=args.learning_rate_step, gamma=args.learning_rate_decay)

def calculate_scores(batch, test=True):
    '''
    batch -> a 4D list containing minibatch of docs.  [doc0, doc1, ..] -> [pdoc0, ndoc0] -> [sent1, sent2, ..] -> [word1, word2, ..]
    pos_batch/neg_batch -> 3d list.  [pdoc0, pdoc1, ..] -> [sent1, sent2, ..] -> [word1, word2, ..]
    batch_docs_len -> 1D list containing len of docs (num of sentences in each docs)
    batch_sentences_len -> 2D list containing original length of each sentences in each docs [doc->len_sent]
    modified_batch_sentences_len -> 2D numpy array containing len of each sentences in each docs after padding  [doc->len_sent]
    '''

    if args.ELMo:
        # docu_batch_idx -> 4D Tensor of char_ids for ELMo model [doc->sentences->word->char_ids]
        docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = utils.batch_preprocessing_elmo(
            batch, args)
    else:
        # docu_batch_idx -> 3D Tensor of word_ids for general embeddings model [doc->sentences->word_ids]
        docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = utils.batch_preprocessing(
            batch, args)
    '''
    output -> 3D Tensor.  [batch_size X doc_max_len, max_sentence_len, 2*args.hidden_dim] 
    hidden -> 3D Tensor.  [batch_size, doc_max_len, 2*args.hidden_dim] 
    '''
    output, hidden = sentence_encoder(
        docu_batch_idx, modified_batch_sentences_len)

    hidden_out = hidden

    ### Global Feature ###
    # make the time dim to first, batch to second - for lightweight conv.  [doc_max_len -> batch_size -> 2*args.hidden_dim]
    hidden_out = hidden_out.permute(1, 0, 2).contiguous()
    # 3D Tensor containing global features from lightweight convolution.  [batch -> 1 -> 2*args.hidden_dim]
    # batch is made first dim in the function
    global_features = global_feature_extractor(hidden_out)
    # hidden_out back to original order.  [batch_size -> doc_max_len -> 2*args.hidden_dim]
    hidden_out = hidden_out.permute(1, 0, 2).contiguous()

    ### Local Feature ###

    # Bilinear layer
    # forward_inputs contain 1 index forward to hidden_out, needed in bilinear_layer
    index = list(range(hidden_out.size(1)))
    index = index[1:]
    index.append(index[-1])
    forw_idx = torch.LongTensor(index).to(
        args.device).requires_grad_(False)
    forward_inputs = torch.index_select(
        hidden_out, dim=1, index=forw_idx)
    # 3D Tensor containing output of bilinear layer.   [doc -> sentence -> bilinear_dim]
    bi_curr_inputs = bilinear_layer(hidden_out, forward_inputs)

    # Linear layer
    # bi_forward_inputs contain 1 index forward to bi_curr_inputs,
    # concat them for linear layer which will give local features of consecutive 2 sentences
    bi_forward_inputs = torch.index_select(
        bi_curr_inputs, dim=1, index=forw_idx)
    # 3D Tensor containing local features of consecutive 2 sentences.
    # [doc -> sentence -> 2*bilinear_dim]
    cat_bioutput_feat = torch.cat(
        (bi_curr_inputs, bi_forward_inputs), dim=2)
    # 3D Tensor containing average values of the local features, needed for calculating loss.
    # [doc -> sentence -> 1]
    mask_val = torch.mean(cat_bioutput_feat, dim=2).unsqueeze(2)
    # 3D Tensor containing global features repeated by #max_sentence.
    # [batch -> sentence -> 2*args.hidden_dim]
    conv_extended = global_features.repeat(
        1, cat_bioutput_feat.size(1), 1)
    # 3D Tensor containing concatenated global+local features.
    # [batch -> sentence -> 2*args.hidden_dim+2*bilinear_dim]
    coherence_feature = torch.cat(
        (cat_bioutput_feat, conv_extended), dim=2)
    # linear layer returns 3D tensor containing scores.   [batch -> sentence -> 1]
    scores = coherence_scorer(coherence_feature)
    # mask value for finding valid scores. valid index contains 1, others 0
    score_mask = utils.score_masked(scores, batch_docs_len, args.device)
    # Only keep the valid scores. 3D tensor containing scores.   [batch -> sentence -> 1]
    masked_score = scores*score_mask

    pos_score = masked_score

    # Document level socre
    # 1D numpy array containing the sum scores of the document.   [batch]
    pos_doc_score = np.asarray([score.sum().data.cpu().numpy()
                                for score in pos_score])

    return pos_doc_score


print(f"Test start...")
with torch.no_grad():
    # best_dir = f"./Models/Best/"
    best_dir = f"./Models/ELMo/"

    model_name = f"Epoch_6_MMdd_6_7"
    # model_name = f"Epoch_11_MMdd_6_6_google"

    model_save_path = os.path.join(
        best_dir, model_name)
    local_global_model.load_state_dict(torch.load(model_save_path, map_location=args.device))
    local_global_model.eval()

    # 3d list.  [pdoc0, pdoc1, ..] -> [sent1, sent2, ..] -> [word1, word2, ..]
    batches = [
        [
            ".START",
            "McDermott International Inc. said its Babcock & Wilcox unit completed the sale of its Bailey Controls Operations to Finmeccanica S.p.",
            "A. for $295 million.",
            "Finmeccanica is an Italian state-owned holding company with interests in the mechanical engineering industry.",
            "Bailey Controls, based in Wickliffe, Ohio, makes computerized industrial controls systems.",
            "It employs 2,700 people and has annual revenue of about $370 million."],
        [
             "McDermott International Inc. said its Babcock & Wilcox unit completed the sale of its Bailey Controls Operations to Finmeccanica S.p.",
             ".START", "A. for $295 million.",
             "Finmeccanica is an Italian state-owned holding company with interests in the mechanical engineering industry.",
             "Bailey Controls, based in Wickliffe, Ohio, makes computerized industrial controls systems.",
             "It employs 2,700 people and has annual revenue of about $370 million."]
    ]

    batch = []
    for doc in batches:
        batch.append([sentence.split() for sentence in doc])

    score = calculate_scores(batch, test=True)

    # Accuracy at a certain Epoch
    print(f"Test Score: {score}")
