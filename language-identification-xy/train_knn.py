from pathlib import Path
import numpy as np

from tqdm import tqdm
from joblib import dump
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)

    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )

    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # %%
    def text_to_ord(text):
        # split text and change chars to utf-8 decimal number
        # delete common chars such as punctuation marks and numbers
        return [ord(char) if ord(char)>64 else 64 for char in text]


    # skip all nan values with np.nanXXX
    text_train["min_char"] = text_train["text"].apply(text_to_ord).apply(np.nanmin)
    text_train["max_char"] = text_train["text"].apply(text_to_ord).apply(np.nanmax)

    # train and save model
    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=3)

    X = pd.concat([text_train["min_char"], text_train["max_char"]], axis=1)
    y = targets_train["lang"]

    neigh.fit(X, y)

    dump(neigh, Path(__file__).parent / "model.joblib")