from pathlib import Path
from typing import Iterable

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def prepare_texts(
    df: pd.DataFrame,
    text_col: str = "text",
    use_lemma: bool = False,
) -> list[str]:
    if use_lemma:
        return (
            df["lemma_clean"]
            .apply(" ".join)
            .tolist()
        )

    return df[text_col].fillna("").tolist()


def run_bertopic(
    df: pd.DataFrame,
    stopwords: Iterable[str],
    text_col: str = "text",
    min_words: int = 100,
    n_subset: int | None = None,
    use_lemma: bool = True,
    min_topic_size: int = 20,
) -> tuple[pd.DataFrame, BERTopic]:

    if n_subset is not None:
        df = df.sample(
            n=n_subset,
            random_state=1512,
        )

    df_topic = df[df["word_count"] >= min_words].copy()

    texts = prepare_texts(
        df_topic,
        text_col=text_col,
        use_lemma=use_lemma,
    )

    vectorizer_model = CountVectorizer(
        stop_words=list(stopwords),
        min_df=10,
        ngram_range=(1, 2),
    )

    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts)

    df_topic["topic"] = topics
    df_topic["topic_probability"] = probs

    return df_topic, topic_model


def save_topic_outputs(
    df_topic: pd.DataFrame,
    topic_model: BERTopic,
    output_dir: str | Path = "data/processed",
    model_dir: str | Path = "models/bertopic_bundestag",
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    topic_info = topic_model.get_topic_info()

    df_topic.to_parquet(
        output_dir / "speeches_with_topics.parquet",
        index=False,
    )

    topic_info.to_parquet(
        output_dir / "topic_info.parquet",
        index=False,
    )

    topic_info.to_csv(
        output_dir / "topic_info.csv",
        index=False,
    )

    topic_model.save(model_dir)


def load_topic_model(
    model_dir: str | Path = "models/bertopic_bundestag",
) -> BERTopic:
    return BERTopic.load(model_dir)

def map_topic_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    TOPIC_LABELS = {
    -1: "Sonstiges",

    0: "Ukraine-Krieg & Russland",
    1: "Migration & Asylpolitik",
    2: "Familie, Kommunen & Bildung",
    3: "Gesundheit & Pflege",
    4: "Digitalisierung & Staatsmodernisierung",
    5: "Klimaschutz & Klimapolitik",
    6: "Landwirtschaft & Agrarpolitik",
    7: "Steuern & Entlastungen",
    8: "Verkehr & Infrastruktur",
    9: "Energiewende & Erneuerbare Energien",
    10: "Parlament & Geschäftsordnung",
    11: "Forschung, Innovation & KI",
    12: "Israel, Gaza & Nahostkonflikt",
    13: "Rente & Altersvorsorge",
    14: "Innere Sicherheit & Polizei",
    15: "Wirtschaft & Wettbewerbsfähigkeit",
    16: "Bundeswehr & Soldaten",
    17: "Entwicklungszusammenarbeit",
    18: "Atomkraft & Kernenergie",
    19: "Bundeshaushalt",
    20: "Energieversorgung & Gaspolitik",
    21: "Wohnen & Mietpolitik",
    22: "Demokratie & Extremismus",
    23: "Europa & Internationale Beziehungen",
    24: "Afghanistan",
    25: "Corona & Pandemieaufarbeitung",
    26: "Wirtschaftspolitik & Investitionen",
    27: "Sportpolitik",
    28: "Auslandseinsätze Mittelmeer & Libyen",
    29: "Queerpolitik & Selbstbestimmungsgesetz",
    30: "Antisemitismus",
    31: "Cybersicherheit",
    32: "China & Handelspolitik",
    33: "Cannabis & Drogenpolitik",
    34: "Islamismus",
    35: "Schwangerschaftsabbruch",
    36: "Verbraucherschutz",
    37: "Schule & Bildung",
    38: "Politischer Schlagabtausch",
    39: "Tierschutz",
    40: "Syrien",
    41: "Sudan",
    42: "Berufliche Bildung",
    43: "Digitale Plattformen & Jugendschutz",
    44: "Familienpolitik & Kindergrundsicherung",
    45: "Mindestlohn & Tarifpolitik",
    46: "Schuldenbremse",
    47: "Bürokratieabbau",
    48: "Deutschlandticket & ÖPNV",
    49: "Energiepreise & Industriestrom",
    50: "Kinderschutz & sexueller Missbrauch",
    51: "Arbeitszeit & Arbeitsmarkt",
    52: "Kinderehen",
    53: "Geldwäsche & Organisierte Kriminalität",
    54: "Luftverkehr",
    55: "Mitbestimmung & Betriebsräte",
    56: "Familiennachzug",
    57: "Demokratieschutz & Strafrecht",
    58: "Ruanda & Völkermord-Gedenken",
    59: "Kulturpolitik",
    60: "Gewalt gegen Frauen",
    61: "Geothermie & Wärmeversorgung",
    62: "Demokratiebeteiligung & Bürgerräte",
    63: "Kosovo-Einsatz",
    64: "Westbalkan-Missionen",
    65: "Islamistischer Terrorismus",
    66: "Mietrecht & Rechtsstaat",
    67: "Batterietechnologie & Northvolt",
    68: "Nachhaltigkeitsstrategie",
    69: "Irak-Einsatz",
    }
    df["topic_label"] = df["topic"].map(TOPIC_LABELS)

    return df

def filter_topics(df: pd.DataFrame, top: int) -> pd.DataFrame:
    return (
        df
        .loc[df["topic"].isin(range(0, top))]
        .copy()
    )