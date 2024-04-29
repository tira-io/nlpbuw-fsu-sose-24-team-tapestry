from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))

    try:
        nlp = spacy.load('en_core_web_md')
    except OSError:
        print('Downloading language model for the spaCy POS tagger')
        from spacy.cli import download
        download('en_core_web_md')
        nlp = spacy.load('en_core_web_md')

    df['word2vec_doc'] = df['text'].apply(lambda text: nlp(text).vector)

    # flattening text representation column from lists into separate columns
    X_train = df['word2vec_doc'].apply(lambda x: pd.Series(x))
    X_train.columns = X_train.columns.astype(str)
            
    model = LogisticRegression()
    model.fit(X_train, df["generated"])
            
    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
