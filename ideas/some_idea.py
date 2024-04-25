from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    text_lens = np.array([len(x) for x in text_train])

    generated_class = targets_train == 1
    written_class = targets_train == 0

    plt.plot(range(len(generated_class)), text_lens[generated_class])
    plt.plot(range(len(generated_class)), text_lens[written_class])

    plt.show()


    '''
    for i in range(10):
        print(text_train)

    # classifying the data
    prediction = (
        text_validation.set_index("id")["text"]
        .str.contains("delve", case=False)
        .astype(int)
    )

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

'''