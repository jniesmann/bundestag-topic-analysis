import spacy
import pandas as pd
from pathlib import Path  


def add_stopwords(
    path_to_stopwords: Path,
    nlp
):

    with open(
        path_to_stopwords,
        "r",
        encoding="utf-8"
    ) as infile:

        for line in infile:
            word = line.strip()

            if not word:
                continue

            nlp.Defaults.stop_words.add(word)
            nlp.vocab[word].is_stop = True

def get_lemma(doc) -> list[str]:
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

def lemmatize(
    df: pd.DataFrame,
    text_column: str,
    nlp
) -> pd.DataFrame:

    all_lemmas = []

    for doc in nlp.pipe(
        df[text_column],
        batch_size=64,
        disable=["ner"]
    ):
        all_lemmas.append(get_lemma(doc))

    df = df.copy()
    df["lemma_clean"] = all_lemmas

    return df

