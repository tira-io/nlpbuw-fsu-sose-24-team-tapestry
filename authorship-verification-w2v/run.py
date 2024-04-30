from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )


    nlp = load(Path(__file__).parent / "en_core_web_md")

    df['word2vec_doc'] = df['text'].apply(lambda text: nlp(text).vector)
    X_val = df['word2vec_doc'].apply(lambda x: pd.Series(x))
    X_val.columns = X_val.columns.astype(str)


    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(X_val)
    df["generated"] = predictions
    df = df[["id", "generated"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
