import sys
import os

sys.path.append('/homes/ml007/works/codes/30-11-2021/finetune_gpt2/src')

from utils import read_data


if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_path = os.path.join(path, "data/amazon_reviews.txt")
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
    split_data = pd_data.sample(frac=0.2, replace=False)
