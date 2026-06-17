import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud


DATA_PATH = Path("data/processed/speeches_with_topics_labeled_red.parquet")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if isinstance(df["lemma_clean"].iloc[0], str):
        df["lemma_clean"] = df["lemma_clean"].apply(ast.literal_eval)

    df["fraction"] = (
        df["fraction"]
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .replace({
            "SPDSPD": "SPD",
            "BÜNDNIS 90/DIE GRÜNEN": "Grüne",
            "Die Linke": "Linke",
        })
    )

    return df


def plot_wordcloud_for_topic(df: pd.DataFrame, label: str):
    topic_df = df[df["topic_label"] == label]
    all_lemmas = topic_df["lemma_clean"].explode().dropna().tolist()
    freqs = Counter(all_lemmas)

    fig, ax = plt.subplots(figsize=(10, 4))

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False,
    ).generate_from_frequencies(freqs)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud: {label}", fontsize=16)

    return fig


def plot_topic_share_over_time(df: pd.DataFrame, label: str):
    topic_df = df[df["topic_label"] == label]

    total_per_month = df.groupby("year_month").size()
    topic_per_month = topic_df.groupby("year_month").size()

    share = (
        topic_per_month
        .div(total_per_month)
        .fillna(0)
        * 100
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    share.plot(
        ax=ax,
        marker="o",
    )

    ax.set_title(f"Anteil der Reden zu: {label}")
    ax.set_ylabel("% aller Reden")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    return fig

def plot_fraction_share(df: pd.DataFrame, label: str):

    PARTY_COLORS = {
        "SPD": "#E3000F",
        "CDU/CSU": "#000000",
        "AfD": "#009EE0",
        "Grüne": "#46962B",
        "FDP": "#FFED00",
        "Linke": "#BE3075",
        "BSW": "#7A003C",
        "fraktionslos": "#808080",
        "unknown": "#B0B0B0",
    }

    topic_df = df[df["topic_label"] == label]

    topic_counts = topic_df["fraction"].value_counts()
    total_counts = df["fraction"].value_counts()

    share = (
        topic_counts
        .div(total_counts)
        .dropna()
        * 100
    ).sort_values()

    colors = [
        PARTY_COLORS.get(fraction, "#808080")
        for fraction in share.index
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    share.plot(
        kind="barh",
        ax=ax,
        color=colors,
    )

    ax.set_title(
        f"Anteil der Fraktionsreden zu: {label}"
    )
    ax.set_xlabel("% der Fraktionsreden")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    return fig


def main():
    st.set_page_config(
        page_title="Bundestag Topic Explorer",
        layout="wide",
    )

    st.title("Bundestag Topic Explorer")

    df = load_data(DATA_PATH)

    labels = (
        df.loc[df["topic"] != -1, "topic_label"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    selected_label = st.sidebar.selectbox(
        "Thema auswählen",
        labels,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Autor: Jan Niesmann")
    st.sidebar.markdown("GitHub: [jniesmann](https://github.com/jniesmann/bundestag-topic-analysis)")

    topic_df = df[df["topic_label"] == selected_label]

    col1, col2, col3 = st.columns(3)

    col1.metric("Reden insgesamt", f"{len(df):,}")
    col2.metric("Topics", df["topic"].nunique())
    col3.metric("Reden im Thema", f"{len(topic_df):,}")

    st.subheader(selected_label)

    tab1, tab2, tab3 = st.tabs(
        [
            "WordCloud",
            "Zeitverlauf",
            "Fraktionen",
        ]
    )

    with tab1:
        st.pyplot(
            plot_wordcloud_for_topic(df, selected_label)
        )

    with tab2:
        st.pyplot(
            plot_topic_share_over_time(df, selected_label)
        )

    with tab3:
        st.pyplot(
            plot_fraction_share(df, selected_label)
        )

    with st.expander("Beispielreden anzeigen"):
        sample_cols = [
            "date",
            "first_name",
            "last_name",
            "fraction",
            "text",
        ]

        examples = (
            topic_df
            .sort_values("topic_probability", ascending=False)
            .head(3)
        )

        for _, row in examples.iterrows():
            st.markdown(
                f"**{row['first_name']} {row['last_name']} "
                f"({row['fraction']})**"
            )
            st.write(row["text"][:1200] + "...")

    with st.expander("Methodik"):
        st.markdown("""
        ### Datenquelle
        Grundlage der Analyse sind die öffentlich verfügbaren XML-Protokolle der Plenardebatten des Deutschen Bundestages aus den Jahren 2024 und 2025.

        Berücksichtigt wurden alle Redebeiträge mit einer Mindestlänge von 100 Wörtern.

        ### Methodik
        Die Redebeiträge wurden zunächst mit spaCy vorverarbeitet. Dies umfasst die Lemmatisierung der Texte sowie die Entfernung allgemeiner deutscher und bundestagsspezifischer Stopwörter.

        Anschließend wurde mittels BERTopic ein Topic Modeling durchgeführt, um thematische Schwerpunkte der Debatten automatisch zu identifizieren. Die resultierenden Topics wurden auf Basis ihrer Schlüsselbegriffe und repräsentativen Redebeiträge manuell benannt.

        Zur besseren Übersicht werden im Dashboard die 35 größten eindeutigen Themencluster dargestellt.

        ### Autor
        Jan Niesmann   
        GitHub: [jniesmann](https://github.com/jniesmann/bundestag-topic-analysis)     
        
        """)


if __name__ == "__main__":
    main()