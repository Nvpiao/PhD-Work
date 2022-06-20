import os
import pandas as pd

from gpt2model.train import seed_everything

if __name__ == '__main__':
    seed_everything(2022)

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

    pd_data = pd.read_csv(data_path, sep='\t', header=None, names=columns, skiprows=1)

    pd_data_20 = pd_data.sample(frac=0.2, replace=False)
    pd_data_80 = pd_data[~pd_data.index.isin(pd_data_20.index)]

    pd_data_20.to_csv("amazon_reviews_20.txt", index=False, sep='\t')
    pd_data_80.to_csv("amazon_reviews_80.txt", index=False, sep='\t')

    print("Done")
