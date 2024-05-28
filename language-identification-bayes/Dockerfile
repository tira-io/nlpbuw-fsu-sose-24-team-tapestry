FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD model_latin.joblib /code/model_latin.joblib
ADD model_cyrillic.joblib /code/model_cyrillic.joblib
ADD vec_latin_vocab.joblib /code/vec_latin_vocab.joblib
ADD vec_cyrillic_vocab.joblib /code/vec_cyrillic_vocab.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]