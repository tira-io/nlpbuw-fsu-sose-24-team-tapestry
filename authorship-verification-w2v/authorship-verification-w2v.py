from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

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

    pd.set_option('display.max_colwidth', 50)
    concat_train = pd.concat([text_train, targets_train['generated']], axis=1)
    concat_val = pd.concat([text_validation, targets_validation['generated']], axis=1)

    try:
        nlp = spacy.load('en_core_web_md')
    except OSError:
        print('Downloading language model for the spaCy POS tagger')
        from spacy.cli import download
        download('en_core_web_md')
        nlp = spacy.load('en_core_web_md')

    for data in (concat_train, concat_val):
        data['word2vec_doc'] = data['text'].apply(lambda text: nlp(text).vector)

    
    model_logistic = LogisticRegression()

    y_train = concat_train.generated
    y_val = concat_val.generated

    # flattening text representation column from lists into separate columns
    X_train = concat_train['word2vec_doc'].apply(lambda x: pd.Series(x))
    X_train.columns = X_train.columns.astype(str)
    X_val = concat_val['word2vec_doc'].apply(lambda x: pd.Series(x))
    X_val.columns = X_val.columns.astype(str)
            
    model_logistic.fit(X_train, y_train)
            
    y_pred = model_logistic.predict(X_val)

    df = pd.DataFrame(y_pred)
    pred_val_df = pd.concat([text_validation, df], axis=1)
    pred_val_df.columns = ["id", "text", "prediction"]

    prediction = (
        pred_val_df.set_index("id")["prediction"]
    )

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
            Path(output_directory) / "predictions_w2v.jsonl", orient="records", lines=True
    )
