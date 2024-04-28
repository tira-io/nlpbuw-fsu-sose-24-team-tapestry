# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD authorship-verification-jupyter.ipynb /code/authorship-verification-jupyter.ipynb

ENTRYPOINT [ "python3", "/code/authorship-verification-jupyter.ipynb" ]
