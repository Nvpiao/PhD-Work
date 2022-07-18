import gradio as gr
import torch
import torch.nn as nn

import numpy as np
import os
import random
from sentence_split import split_into_sentences

import sys
sys.path.append('../../')

from finetune_gpt2.src.gpt2model_with_first_last.generator \
    import load_tokenizer, load_pretrained_mode
from finetune_gpt2.src.gpt2model_with_first_last.train \
    import seed_everything

from unified_coherence_model.src \
    import utils, model
from unified_coherence_model.test_unified_coherence \
    import calculate_scores

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

random.seed(2022)

GPT2_FINETUNE_MODEL = None
GPT2_FINETUNE_TOKENIZER = None
COHERENCE_RANKER_MODEL = None
MAX_LEN = 100
NUM_RETURN_SEQ = 10
SENTENCES_LEN = 2

special_tokens = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                  "mask_token": "[MASK]",
                  "bos_token": "[BOS]", "eos_token": "[EOS]"}


def get_gpt2_finetune_model():
    path = os.path.abspath(os.path.join(os.getcwd(), f"../../finetune_gpt2"))
    model_path = os.path.join(path, f"model/gpt2_finetune_first_last_all_20/")

    # path = os.path.abspath(os.path.join(os.getcwd(), "..\\.."))
    # model_path = os.path.join(path, "model\\gpt2_finetune_pr_rt_rt/")

    tokenizer = load_tokenizer(model_path)
    model = load_pretrained_mode(tokenizer, model_path, special_tokens)

    return model, tokenizer


def get_coherence_ranker_model():
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

    parent_path = f'../../unified_coherence_model/'

    best_dir = parent_path + f"./Models/Best/"
    # best_dir = parent_path + f"./Models/ELMo/"

    model_name = f"Epoch_11_MMdd_6_6_google"
    # model_name = f"Epoch_6_MMdd_6_7"

    model_save_path = os.path.join(
        best_dir, model_name)
    local_global_model.load_state_dict(torch.load(model_save_path, map_location=args.device))

    return local_global_model


def calculate_ranker(first, middles, last):
    batch = []
    for sentence in middles:
        batch.append([sentence.split() for sentence in [first, sentence, last]])

    score = calculate_scores(batch, test=True)
    return np.argmax(score)


def get_gradio_fn(first_sentence, last_sentence, max_len, num_return_sequences, sentence_len):
    first_sentence = first_sentence.strip()
    last_sentence = last_sentence.strip()

    max_len = int(max_len)
    num_return_sequences = int(num_return_sequences)
    if GPT2_FINETUNE_MODEL is not None \
            and GPT2_FINETUNE_TOKENIZER is not None \
            and COHERENCE_RANKER_MODEL is not None:
        seed_everything(2022)

        # evaluation model
        GPT2_FINETUNE_MODEL.eval()
        COHERENCE_RANKER_MODEL.eval()

        sentence_res = [first_sentence, last_sentence]
        while sentence_len > 0:
            sentence_index = 1
            sentence_can = []
            for i in range(0, len(sentence_res) - 1):
                sentence_can.append([sentence_res[i], sentence_res[i + 1]])

            for first, last in sentence_can:
                first = first.strip()
                last = last.strip()
                input_text = special_tokens['bos_token'] + first \
                             + special_tokens['sep_token'] + last + special_tokens['sep_token']
                input_text_encoder = torch.tensor(GPT2_FINETUNE_TOKENIZER.encode(input_text)).unsqueeze(0)
                GPT2_FINETUNE_MODEL.to(args.device)
                input_text_encoder = input_text_encoder.to(args.device)

                generated_res = GPT2_FINETUNE_MODEL.generate(input_text_encoder,
                                                             do_sample=True,
                                                             min_length=50,
                                                             max_length=max_len,
                                                             top_k=30,
                                                             top_p=0.7,
                                                             temperature=0.9,
                                                             repetition_penalty=2.0,
                                                             num_return_sequences=num_return_sequences)

                # first sentence of top-K results
                generated_sens = []
                for i, output in enumerate(generated_res):
                    text = GPT2_FINETUNE_TOKENIZER.decode(output, skip_special_tokens=True)
                    text_len = len(first) + len(last)
                    # TODO  fix null pointer bug
                    generated_sens.append(split_into_sentences(text[text_len:])[0].strip())

                #print('generated_sentences: ', "\r\n".join(generated_sens), '\r\n')
                # calculate the rank of each sentence and return the index with highest score.
                highest_score_index = calculate_ranker(first, generated_sens, last)

                sentence_res.insert(sentence_index, generated_sens[highest_score_index])
                sentence_index += 2

            sentence_len -= 1
        return " ".join(sentence_res)
    else:
        return "Error occur when initializing the generator !"


if __name__ == "__main__":
    # set seed
    seed_everything(2022)

    # init gpt2 model
    GPT2_FINETUNE_MODEL, GPT2_FINETUNE_TOKENIZER = get_gpt2_finetune_model()

    # init coherence ranker
    COHERENCE_RANKER_MODEL = get_coherence_ranker_model()

    # gradio
    gradio_model = gr.Interface(
        fn=get_gradio_fn,
        inputs=[
            gr.inputs.Textbox(lines=2, default="", label="First Sentence"),
            gr.inputs.Textbox(lines=2, default="", label="Last Sentence"),
            gr.inputs.Number(default=MAX_LEN, label="Max Len"),
            gr.inputs.Number(default=NUM_RETURN_SEQ, label="Num Of Candidate Sentence"),
            gr.inputs.Number(default=SENTENCES_LEN, label="Iteration times (1:3s, 2:5s, 3:9s)"),
        ],
        outputs="text"
    )
    # gradio_model.launch()
    gradio_model.launch(share=True)
