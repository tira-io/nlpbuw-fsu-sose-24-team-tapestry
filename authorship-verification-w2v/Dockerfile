# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD authorship-verification-w2v.py /code/authorship-verification-w2v.py

ENTRYPOINT [ "python3", "/code/authorship-verification-w2v.py" ]
