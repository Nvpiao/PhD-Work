import argparse
import datetime
import json
import pickle
import torch
import numpy as np

import sys
sys.path.append(f'../')

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str,
                        default="./Models/ELMo/", help='Save path of best model')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='save best model?')
    parser.add_argument('--n_window', type=int, default=3,
                        help='Number of permutation window. Only for local. values: 1/2/3')
    parser.add_argument('--train_path', type=str,
                        default="./Data/Amazon/Dataset/train/", help='Train paired Data')  # "../data-global/train/"
    parser.add_argument('--test_path', type=str,
                        default="./Data/Amazon/Dataset/test/", help='test/Dev paired Data')
    parser.add_argument('--file_list_train', type=str, default="./Data/Amazon/wsj.train",
                        help='Only for Global Dataset: Train Data list')
    parser.add_argument('--file_list_test', type=str, default="./Data/Amazon/wsj.dev",
                        help='Only for Global Dataset: test/Dev Data list')
    parser.add_argument('--pre_embedding_path', type=str,
                        default="./Models/GoogleNews/GoogleNews-vectors-negative300.bin", help='Pretrained word embedding path')
    parser.add_argument('--vocab_path', type=str,
                        default="./Data/Amazon/Dataset/vocab/Vocab", help='Vocab path')
    parser.add_argument('--padding_symbol', type=str,
                        default="<pad>", help='Vocab path')
    # Training Parameter-------------------------------------------------------------
    parser.add_argument('--Epoch', type=int, default=25,
                        help='Number of Epoch ')
    parser.add_argument('--learning_rate_step', type=int, default=5,
                        help='Decrease learning rate for every certain epoch ')
    parser.add_argument('--learning_rate_decay', type=float, default=.1,
                        help='Decrease learning rate for every certain epoch ')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Decrease learning rate for every certain epoch ')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='Optimizer learning rate')
    parser.add_argument('--ranking_loss_margin', type=float,
                        default=5, help='ranking loss margin')
    parser.add_argument('--device', type=str, default='cuda', help='CPU? GPU?')
    # Minibatch argument
    parser.add_argument('--batch_size_train', type=int,
                        default=4, help='Mini batch size')
    parser.add_argument('--batch_size_test', type=int,
                        default=4, help='Mini batch size for test/dev')
    parser.add_argument('--shuffle', type=bool,
                        default=True, help='shuffle items')
    parser.add_argument('--file_type', type=str,
                        default='json', help='Load file type')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Only for Local Datasets: Local window size')
    # Network Parameter
    parser.add_argument('--n_vocabs', type=int,
                        help='Word embedding dim, it should be defined using the vocab list')
    parser.add_argument('--embed_dim', type=int,
                        default=300, help='Word embedding dim')
    # RNN Parameter
    parser.add_argument('--hidden_dim', type=int,
                        default=256, help='Hidden dim of RNN')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout ratio of RNN')
    parser.add_argument('--bidirectional', type=bool,
                        default=True, help='Bi-directional RNN?')
    parser.add_argument('--batch_first', type=bool,
                        default=True, help='Dimension order')
    # Light-weight convolution Parameters
    parser.add_argument('--num_head', type=int, default=16,
                        help='Number of heads in DyConv')
    parser.add_argument('--kernel_size', type=int,
                        default=5, help='Kernel size of DyConv')
    parser.add_argument('--conv_dropout', type=float,
                        default=.0, help='DyConv kernel dropout rate')
    parser.add_argument('--kernel_padding', type=int,
                        default=3, help='DyConv kernel padding')
    parser.add_argument('--kernel_softmax', type=bool,
                        default=True, help='DyConv kernel softmax')

    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument('--GoogleEmbedding', type=bool,
                           default=True, help='Google embedding')
    embedding.add_argument('--RandomEmbedding', type=bool,
                           default=False, help='Random embedding')
    embedding.add_argument('--ELMo', type=bool,
                           default=False, help='ELMo embedding')
    parser.add_argument('--ELMo_Size', type=str,
                        default='small', help='Size of ELMo')
    parser.add_argument('--bilinear_dim', type=int,
                        default=32, help='bilinear output dim')
    parser.add_argument('--lm_loss_weight', type=float, default=1.0,
                        help='weight of Language Model loss to be counted in final loss')
    parser.add_argument('--dataset', type=str, default='data-global',
                        help='Which data-set? Options: data-tokenized, data-full, data-global')
    parser.add_argument('--eval_task', type=str, default='std',
                        help='2 discrimination tasks: std-> standard, inv->inverse')
    parser.add_argument('--global_model', type=bool,
                        default=True, help='Whether to add global model with local model')

    return parser


def print_args(args):
    """
    Print all arguments in argparser
    """
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    now = datetime.datetime.now()
    print(f"||Experiment Date:{now.year}-{now.month}-{now.day}||")
    print("Arguments List: \n")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")

    print("---------------------------------------------------------")
    print("---------------------------------------------------------\n")


def load_file(path, file_type):
    if file_type == 'pickle':
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        return data
    elif file_type == 'json':
        with open(path, 'r') as fout:
            data = json.load(fout)
        return data
    elif file_type == 'npy':
        data = np.load(path)
        return data
    else:
        print("File Type Error")


def unpairing_pos_neg(doc_batch):
    '''
    doc_batch -> 4d list containing minibatch of docs
    <==> retruns two 3d lists.  [pdoc0, pdoc1, ..] -> [sent1, sent2, ..] -> [word1, word2, ..]
    '''
    pos_batch = []
    neg_batch = []
    for pos_neg in doc_batch:
        pos_batch.append(pos_neg[0])
        neg_batch.append(pos_neg[1])
    return pos_batch, neg_batch


def batch2idx(batch, word2idx):
    """
    batch -> 3d list of words    [doc->sentence->words]
    word2idx -> dictionary of word2idx
    <==> returns 3d list of word ids  [doc->sentence->word_ids]
    """
    batch_list = []
    for doc in batch:
        doc_list = []
        for sent in doc:
            sent_list = []
            for word in sent:
                try:
                    idx = word2idx[word]
                    sent_list.append(idx)
                except:
                    idx = word2idx['<unk>']
                    sent_list.append(idx)
            doc_list.append(sent_list)
        batch_list.append(doc_list)
    return batch_list


def batch_sentences_length(batch_idx):
    '''
    batch_idx -> 3d list of word ids  [doc->sentence->word_ids]
    <==> returns 2d list of sentence length of each docs  [doc->sent_len]
    '''
    batch_sentences_len = []
    for i in range(len(batch_idx)):
        sentences_len = [len((batch_idx[i])[j])
                         for j in range(len(batch_idx[i]))]
        batch_sentences_len.append(sentences_len)
    return batch_sentences_len


def pad_sentences(docs, docs_len, pad_idx):
    '''
    docs -> 3d list of word_ids  [doc->sentence->word_ids]
    docs_len -> 2d list of sentence length of each docs  [doc->sent_len]
    pad_idx -> id of <pad>
    <==> makes all the doc same length (#sentences) with padding
    '''
    maxlen = max([max(docs_len[i]) for i in range(len(docs_len))])

    def padding(i, j):
        # doc i, sentence j
        needed = maxlen - len((docs[i])[j])
        return (docs[i])[j]+[pad_idx for k in range(needed)]

    return [[padding(i, j) for j in range(len(docs[i]))] for i in range(len(docs))]


def pad_docs(batch_idx_padded, pad_idx):
    """
    batch_idx_padded -> 3d list of word_ids  [doc->sentence->word_ids] 
    pad_idx -> id of <pad>
    <==> Docs that are shorter than the max_doc_len will be padded
    """
    padded_batch = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in batch_idx_padded], batch_first=True, padding_value=pad_idx)
    return padded_batch


def padded_batch_senteces_len(X, maxlen):
    '''
    X -> 2D list containing len of each sentences in each docs [doc->len_sent]
    maxlen -> In padded_batch, every sentences are of same length which is equal to maxlen
    <==> returns modified_batch_sentences_len where original batch_sentences_len is appended by padded (by 1 as padding by <pad> idx is done)
    '''
    new_X = np.ones((len(X), maxlen), dtype=int)
    for i in range(len(X)):
        if (len(X[i]) > maxlen):
            new_X[i] = X[i][0:maxlen]
        else:
            new_X[i][0:len(X[i])] = X[i]
    return new_X


def batch_preprocessing(batch, args):
    """
    batch -> 3d list of words, not indices.  [pdoc0, pdoc1, ..] -> [sent1, sent2, ..] -> [word1, word2, ..]  
    """
    batch_idx = batch2idx(
        batch, args.word2idx)  # 3d list of word_ids  [doc->sentence->word_ids]
    # 1D list containing len of docs (num of sentences in each docs)
    batch_docs_len = [len(batch_idx[i]) for i in range(len(batch_idx))]
    # 2D list containing len of each sentences in each docs [doc->len_sent]
    batch_sentences_len = batch_sentences_length(batch_idx)
    # makes every sentences same lengths (3d list of word_ids)
    batch_idx_padded = pad_sentences(
        batch_idx, batch_sentences_len, args.padding_idx)
    # making every doc same length. 3D Tensor [doc->sentences->word_ids]
    padded_batch = pad_docs(
        batch_idx_padded, args.padding_idx)
    # 2D numpy array containing len of each sentences in each docs after padding
    modified_batch_sentences_len = padded_batch_senteces_len(
        batch_sentences_len, padded_batch.size(1))
    return padded_batch.to(args.device), batch_docs_len, batch_sentences_len, modified_batch_sentences_len

def order_creator_standard(pos_batch, neg_batch, batch_docs_len, device):
    '''
    create negative doc order for standard discrimination task.
    Here neg doc is the permutation of the pos doc i.e. just order of sentences are different.
    Finds out the sentence order of pos doc in neg doc
    '''
    label = []
    max_len = max(batch_docs_len)
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = []
        tracker = np.zeros(len(pos_doc))
        for i, (pos, neg) in enumerate(zip(pos_doc, neg_doc)):
            if pos != neg:
                for j in range(len(pos_doc)):
                    if pos_doc[j] == neg and tracker[j] == 0:
                        label_temp.append(j)
                        tracker[j] = 1
                        break
            else:
                label_temp.append(i)
        if len(label_temp) < max_len:
            pad_len = max_len - len(label_temp)
            for pad in range(pad_len):
                label_temp.append(i+pad+1)
        label.append(torch.LongTensor(label_temp).to(device))
    return label


def order_creator_inverse(pos_batch, neg_batch, batch_docs_len, device):
    '''
    create negative doc order for inverse.
    inverse neg doc is just reverse of pos doc.
    padded portion is done normally. 
    suppose pos doc len is 4. then inverse sentence order [3,2,1,0]. 
    if padded doc len is 10, then label would be [3,2,1,0,4,5,6,7,8,9]
    '''
    label = []
    max_len = max(batch_docs_len)
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = [i for i in range(len(pos_doc))][::-1]

        if len(label_temp) < max_len:
            pad_len = max_len - len(label_temp)
            for pad in range(pad_len):
                # label_temp.append(i+pad+1)
                label_temp.append(len(pos_doc)+pad)
        label.append(torch.LongTensor(label_temp).to(device))
    return label


def score_masked(scores, batch_docs_len, device):
    '''
    returns mask value for finding valid scores. valid index contains 1, others 0
    '''
    score_mask = torch.ones_like(scores).to(device).requires_grad_(False)
    for i, valid_length in enumerate(batch_docs_len):
        # subtracting 2 because for each doc we are getting output of window size 3.
        # For example: if doc has 5 sentences, our model will predict score for
        # 3 ==>[1,2,3],[2,3,4],[3,4,5]
        score_mask[i, valid_length-2:, :] = 0
    return score_mask
