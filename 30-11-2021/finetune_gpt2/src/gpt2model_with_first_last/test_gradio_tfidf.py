import heapq

import gradio as gr
import os
import torch
import joblib

import sys
sys.path.append('/homes/ml007/works/codes/30-11-2021/finetune_gpt2/src')

# from generator import seed_everything, load_tokenizer, load_pretrained_mode
from gpt2model_with_first_last.generator import load_tokenizer, load_pretrained_mode
from gpt2model_with_first_last.train import seed_everything

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'

GPT2_FINETUNE_MODEL = None
GPT2_FINETUNE_TOKENIZER = None
TFIDF_MODEL = None
TFIDF_WORDS = None
TOPK = 10
MAX_LEN = 100
NUM_RETURN_SEQ = 5

special_tokens = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
                  "mask_token": "[MASK]",
                  "bos_token": "[BOS]", "eos_token": "[EOS]"}


def get_gpt2_finetune_model():
    path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    model_path = os.path.join(path, "model/gpt2_finetune_first_last_all_10/")

    # path = os.path.abspath(os.path.join(os.getcwd(), "..\\.."))
    # model_path = os.path.join(path, "model\\gpt2_finetune_pr_rt_rt/")

    tokenizer = load_tokenizer(model_path)
    model = load_pretrained_mode(tokenizer, model_path, special_tokens)
    tfidf_model = joblib.load('../../model/gpt2_finetune_first_last_all_10/.pkl')
    words = tfidf_model.get_feature_names()

    return model, tokenizer, tfidf_model, words


def get_gradio_fn(product_title, review_text, max_len, num_return_sequences):
    max_len = int(max_len)
    num_return_sequences = int(num_return_sequences)
    if GPT2_FINETUNE_MODEL is not None and GPT2_FINETUNE_TOKENIZER is not None:
        seed_everything(2022)

        # tfidf
        tfidf_review = TFIDF_MODEL.transform([review_text]).toarray()[0]
        index = heapq.nlargest(TOPK, range(len(tfidf_review)), tfidf_review.__getitem__)
        review_keywords = ""
        for j in range(len(index)):
            if tfidf_review[index[j]] != 0:
                review_keywords += ' ' + TFIDF_WORDS[index[j]]

        input_text = special_tokens['bos_token'] + product_title \
                     + special_tokens['sep_token'] + review_keywords + special_tokens['sep_token']
        input_text_encoder = torch.tensor(GPT2_FINETUNE_TOKENIZER.encode(input_text)).unsqueeze(0)
        GPT2_FINETUNE_MODEL.to(DEVICE)
        input_text_encoder = input_text_encoder.to(DEVICE)
        # evaluation model
        GPT2_FINETUNE_MODEL.eval()
        generated_res = GPT2_FINETUNE_MODEL.generate(input_text_encoder,
                                                     do_sample=True,
                                                     min_length=50,
                                                     max_length=max_len,
                                                     top_k=30,
                                                     top_p=0.7,
                                                     temperature=0.9,
                                                     repetition_penalty=2.0,
                                                     num_return_sequences=num_return_sequences)

        generated_sens = "Keywords: " + review_keywords + '\n\n'
        for i, output in enumerate(generated_res):
            text = GPT2_FINETUNE_TOKENIZER.decode(output, skip_special_tokens=True)
            text_len = len(product_title) + len(review_keywords)
            generated_sens += 'Generated Sentences ' + str(i + 1) + ': ' + text[text_len - 1:] + '\r\n\r\n'
        return generated_sens
    else:
        return "Error occur when initializing the generator !"


if __name__ == "__main__":
    # set seed
    seed_everything(2022)

    # init model
    GPT2_FINETUNE_MODEL, GPT2_FINETUNE_TOKENIZER, \
        TFIDF_MODEL, TFIDF_WORDS = get_gpt2_finetune_model()

    # gradio
    gradio_model = gr.Interface(
        fn=get_gradio_fn,
        inputs=[
            gr.inputs.Textbox(lines=2, default="", label="Product Title"),
            gr.inputs.Textbox(lines=2, default="", label="Review Text"),
            gr.inputs.Number(default=MAX_LEN, label="Max Len"),
            gr.inputs.Number(default=NUM_RETURN_SEQ, label="Num Of Returned Sequences"),
        ],
        outputs="text"
    )
    # gradio_model.launch()
    gradio_model.launch(share=True)
