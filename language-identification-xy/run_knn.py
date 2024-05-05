from pathlib import Path
import numpy as np

from tqdm import tqdm
from joblib import dump
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"language-identification-validation-20240429-training"
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    # TODO: apply preproc
    predictions = model.predict(df["text"])
    df["lang"] = predictions
    df = df[["id", "lang"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )