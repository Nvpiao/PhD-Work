import os

import pandas as pd

DATA_TYPE_1 = '__label1__'
DATA_TYPE_2 = '__label2__'


def read_data(data_path, columns):
    datasets = pd.read_csv(data_path, sep='\t', header=None, names=columns)
    return datasets


def split_data(data, split_ratio, data_type=DATA_TYPE_1):
    """
        split data by type
    """
    data_type_1 = data[data['LABEL'] == data_type]
    data_type_2 = data[data['LABEL'] != data_type]

    train_set = data.sample(frac=split_ratio, replace=False)
    test_set = data[~data.index.isin(train_set.index)]

    train_set_type_1 = data_type_1.sample(frac=split_ratio, replace=False)
    test_set_type_1 = data_type_1[~data_type_1.index.isin(train_set_type_1.index)]

    train_set_type_2 = data_type_2.sample(frac=split_ratio, replace=False)
    test_set_type_2 = data_type_2[~data_type_2.index.isin(train_set_type_2.index)]

    return train_set, test_set, train_set_type_1, test_set_type_1, train_set_type_2, test_set_type_2


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_path = os.path.join(path, "data\\amazon_reviews.txt")
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

    pd_data = read_data(data_path, columns)

    train_set, test_set, \
    train_set_type_1, test_set_type_1, \
    train_set_type_2, test_set_type_2 = split_data(pd_data, 0.9)
    # print(pd_data.head(10))
