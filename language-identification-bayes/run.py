from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np

def get_block(*ranges):
    block = []
    for r in ranges:
        r = r.split('-')
        block += list(range(int(r[0], 16), int(r[1], 16) + 1))
    return block

def comp_freq(text, block):
    encoded = np.array([ord(c) for c in text])
    return np.sum(np.isin(encoded, block)) / len(encoded)

freq_vec = np.vectorize(comp_freq, excluded={1})

def is_latin(texts):
    latin_block = get_block('0041-024F')
    freqs = freq_vec(texts, latin_block)
    return freqs > 0.5

def is_cyrillic(texts):
    cyrillic_block = get_block('0400-04FF', '0500-052F')
    freqs = freq_vec(texts, cyrillic_block)
    return freqs > 0.5

remove_punctuation = str.maketrans('', '', r"-()\"#/@;:<>{}-=~|.?,")

def PunctFreeLower(texts):
    cleaned = []
    for text in texts:
        cleaned.append((re.sub(r"[0-9]+", "", (text.translate(remove_punctuation)))).lower())
    return cleaned

def classify_remainders(texts):
    langs = np.array(list(non_latin_blocks.keys()))
    freqs = np.empty(shape=(texts.shape[0], len(langs)))
    for i, lang in enumerate(langs):
        block = get_block(non_latin_blocks[lang])
        freqs[:, i] = freq_vec(texts, block)
    preds = langs[np.argmax(freqs, axis=1)]
    return preds


# Load the data
tira = Client()
text = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
)
#text = text.set_index("id")
labels = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
)

df = text.merge(labels, how='left')
df["pred_lang"] = pd.Series([""] * 320000, index=df.index)

# Split texts in latin and non-latin languages
pred_latin = is_latin(df["text"])


# Classify all latin-languages
text_latin = df.loc[pred_latin, ('text')]
text_latin_cleaned = PunctFreeLower(text_latin)

vocab = load(Path(__file__).parent / "vec_latin_vocab.joblib")
vec = CountVectorizer(vocabulary=vocab)
text_latin_vec = vec.fit_transform(text_latin_cleaned)
   
# Load the latin model and make predictions
model = load(Path(__file__).parent / "model_latin.joblib")
predictions = model.predict(text_latin_vec)
df.loc[pred_latin, ('pred_lang')] = predictions


# Classify cyrillic languages
pred_cyrillic = is_cyrillic(text['text'])
text_cyrillic = df.loc[pred_cyrillic, ('text')]
text_cyrillic_cleaned = PunctFreeLower(text_cyrillic)

vocab = load(Path(__file__).parent / "vec_cyrillic_vocab.joblib")
vec = CountVectorizer(vocabulary=vocab)
text_cyrillic_vec = vec.fit_transform(text_cyrillic_cleaned)
   
# Load the cyrillic model and make predictions
model = load(Path(__file__).parent / "model_cyrillic.joblib")
predictions = model.predict(text_cyrillic_vec)
df.loc[pred_cyrillic, ('pred_lang')] = predictions

# Classify remainders
non_latin_blocks = {'el': '0370-03FF', 'zh': '4E00-9FFF', 'ko': 'AC00-D7AF', 'ur': '0600-06FF'}

pred_remainders = ~(pred_latin | pred_cyrillic)
text_remainders = df.loc[pred_remainders, ('text')] # all non-latin, non-cyrillic texts

lang_remainders = classify_remainders(text_remainders)
df.loc[pred_remainders, ('pred_lang')] = lang_remainders


# Save the predictions
df_ = df.loc[:, ('id', 'pred_lang')]
output_directory = get_output_directory(str(Path(__file__).parent))
df_.to_json(
    Path(output_directory) / "predictions.jsonl", orient="records", lines=True
)
