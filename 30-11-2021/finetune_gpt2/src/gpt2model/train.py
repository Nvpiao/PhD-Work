import argparse
import os
import random

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TrainingArguments, Trainer

import sys
sys.path.append('/homes/ml007/works/codes/30-11-2021/finetune_gpt2/src')

from gpt2model.dataset import AmazonDataset, get_train_val_dataloader
from utils import read_data, split_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = 'cpu'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    """
    set seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_tokenizer(tokenizer_path, special_tokens=None):
    """
    load tokenizer
    :param tokenizer_path:
    :param special_tokens:
    :return:
    """
    print('tokenizer loading...')
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    if special_tokens:
        print('special tokens loading...')
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def load_pretrained_mode(tokenizer, pretrained_model_path, special_tokens=None):
    """
    load pretrained model
    :param tokenizer:
    :param pretrained_model_path:
    :param special_tokens:
    :return:
    """
    print("pretrained model loading...")
    gpt2_config = GPT2Config.from_pretrained(pretrained_model_path,
                                             bos_token_id=tokenizer.bos_token,
                                             eos__token_id=tokenizer.eos_token,
                                             sep_token_id=tokenizer.sep_token,
                                             pad_token_id=tokenizer.pad_token,
                                             output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_path, config=gpt2_config)

    if special_tokens:
        # if special tokens are added,model embedding size need to be resized
        model.resize_token_embeddings(len(tokenizer))

    """
    # bias and layernorm.weight non-attenuation
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    """

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # 1.only train last six blocks
    '''
    for i, m in enumerate(model.transformer.h):
        if (i + 1) > 6:
            for param in m.parameters():
                param.requires_grad=True
    '''
    # 2. only train the last layer
    for param in model.lm_head.parameters():
        param.requires_grad = True

    return model.to(DEVICE)


def train_val(model, tokenizer, train_dataset, val_dataset, param_args):
    """
    train model
    :param model:
    :param tokenizer:
    :param train_dataset
    :param val_dataset
    :param param_args
    :return:
    """

    training_args = TrainingArguments(output_dir=param_args.output_dir,
                                      num_train_epochs=param_args.epochs,
                                      per_device_train_batch_size=param_args.batch_size,
                                      per_device_eval_batch_size=len(val_dataset),
                                      gradient_accumulation_steps=param_args.gradient_accumulation_steps,
                                      evaluation_strategy=param_args.evaluation_strategy,
                                      fp16=param_args.fp16,
                                      fp16_opt_level=param_args.apex_opt_level,
                                      warmup_steps=param_args.warmup_steps,
                                      learning_rate=param_args.lr,
                                      adam_epsilon=param_args.adam_eps,
                                      weight_decay=param_args.weight_decay,
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      logging_dir=param_args.logging_dir,
                                      )
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), "..\\.."))
    print("path : {}".format(path))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, "model\\pre_trained_model_gpt2"),
        type=str,
        required=False,
        help='path of pre-trained model'
    )
    parser.add_argument(
        "--config_path",
        default=os.path.join(path, "model\\pre_trained_model_gpt2\\config.json"),
        type=str,
        required=False,
        help="model parameters",
    )
    parser.add_argument(
        '--special_token_path',
        default=os.path.join(path, 'model\\pre_trained_model_gpt2\\special_tokens_map.json')
    )
    parser.add_argument(
        "--vocab_path",
        default=os.path.join(path, "model\\pre_trained_model_gpt2\\vocab.json"),
        type=str,
        required=False,
        help="path of vocab",
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(path, 'data\\amazon_reviews.txt'),
        type=str,
        required=False,
        help="path of train sets",
    )
    parser.add_argument("--epochs", default=10, type=int, required=False, help="train epochs")
    parser.add_argument(
        "--batch_size", default=1, type=int, required=False, help="train batch size"
    )
    parser.add_argument("--lr", default=1.5e-3, type=float, required=False, help="learning rate")
    parser.add_argument("--warmup_steps", default=1e2, type=float, required=False, help="lr warmup steps")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="gradient accumulation steps")
    parser.add_argument("--weight_decay", default=1e-2, type=float, required=False, help="weight decay")
    parser.add_argument(
        "--max_length", default=768, type=int, required=False, help="max length of a sentence"
    )
    parser.add_argument(
        "--train_ratio", default=0.9, type=float, required=False, help="tran and test ratio"
    )
    parser.add_argument(
        "--print_loss", default=1, type=int, required=False, help="steps of printing training loss"
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(path, 'model\\gpt2_finetune_pr_rt_rt'), type=str, required=False,
        help="path of output model"
    )
    parser.add_argument("--logging_dir", default=os.path.join(path, 'model\\gpt2_finetune_pr_rt_rt\\logs'),
                        type=str, required=False, help="log dir")
    parser.add_argument(
        "--seed", default=2022, type=int, required=False, help="python hash seed"
    )
    parser.add_argument(
        "--use_apex", default=True, type=bool, required=False, help="use apex"
    )
    parser.add_argument("--fp16", default=True, type=bool, required=False, help="use float16")
    parser.add_argument("--evaluation_strategy", default="steps", type=str, required=False, help="evaluation strategy")
    parser.add_argument("--adam_eps", default=1e-8, type=float, required=False, help="avoid dividing 0")
    parser.add_argument("--apex_opt_level", default="o1", type=str, required=False, help="apex train type")

    args = parser.parse_args()

    pretrained_model_path = args.pretrained_model_path
    config_path = args.config_path
    vocab_path = args.vocab_path
    data_path = args.data_path
    special_token_path = args.special_token_path
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    max_length = args.max_length
    train_ratio = args.train_ratio
    print_loss = args.print_loss
    output_dir = args.output_dir
    logging_dir = args.logging_dir
    seed = args.seed
    use_apex = args.use_apex
    apex_opt_level = args.apex_opt_level
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    fp16 = args.fp16
    evaluation_strategy = args.evaluation_strategy

    special_tokens = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                      "mask_token": "[MASK]",
                      "bos_token": "[BOS]", "eos_token": "[EOS]"}

    # train data format
    columns = [
        'DOC_ID',
        'LABEL',
        'RATING',
        'VERIFIED_PURCHASE',
        'PRODUCT_CATEGORY',
        'PRODUCT_ID',
        'PRODUCT_TITLE',
        'REVIEW_TITLE',
        'REVIEW_TEXT'
    ]

    seed_everything(seed)

    # read data
    pd_data = read_data(data_path, columns)

    # split train and test
    train_set, test_set, _, _, _, _ = split_data(pd_data, train_ratio)
    # train_set, test_set, \
    # train_set_type_1, test_set_type_1, \
    # train_set_type_2, test_set_type_2 = split_data(pd_data, train_ratio)

    # load tokenize
    tokenizer = load_tokenizer(pretrained_model_path, special_tokens)

    # create datasets
    train_set = AmazonDataset(train_set, tokenizer, max_length, special_tokens)
    # test_set = AmazonDataset(test_set, tokenizer, max_length, special_tokens)
    _, _, train_dataset, val_dataset = get_train_val_dataloader(batch_size, train_set, train_ratio)

    # load pretrained model and fine tune
    model = load_pretrained_mode(tokenizer, pretrained_model_path, special_tokens)

    # build model,no pretrained model
    # model = build_mode(tokenizer, config_path, special_tokens)

    # train and val
    train_val(model, tokenizer, train_dataset, val_dataset, args)
