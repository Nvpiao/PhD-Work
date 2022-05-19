import random

import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize

class AmazonDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, special_tokens, randomize=True):
        """
        create dataset
        :param data: data frame
        :param tokenizer:
        :param max_length:
        :param special_tokens:
        :param randomize:
        """
        self.data = data
        self.data_size = len(data)
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.randomize = randomize
        self.max_length = max_length

    def __getitem__(self, idx):
        row_obj = self.data.iloc[idx]
        review_text = row_obj['REVIEW_TEXT']
        sent_tokens = sent_tokenize(review_text)
        #
        # keywords = review_title.split()
        # keywords = self.join_keywords(keywords, self.randomize)

        inputs = self.special_tokens['bos_token'] + sent_tokens[0] \
                + self.special_tokens['sep_token'] + sent_tokens[-1] \
                + self.special_tokens['sep_token'] + review_text \
                + self.special_tokens['eos_token']

        encodings_dict = self.tokenizer(inputs,
                                        truncation=True,
                                        max_length=self.max_length,
                                        padding='max_length')
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}

    def __len__(self):
        return self.data_size


def get_train_val_dataloader(batch_size, train_set, train_ratio):
    """
    split train set into train and validation sets
    :param batch_size:
    :param train_set:
    :param train_ratio
    :return: train\\validation datasets and loaders
    """

    train_size = int(train_ratio * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=len(val_dataset),
                            shuffle=False, )

    return train_loader, val_loader, train_dataset, val_dataset
