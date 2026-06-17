from bt_pipeline import data_loader
from bt_pipeline import text_preprocessing, xml_parser, topic_modeling
import argparse       
import spacy         
from pathlib import Path  
import pandas as pd 
import os


def main_analysis():

    parser = argparse.ArgumentParser(description="Bundestags-Topic-Modeling Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Überspringt das Herunterladen der XML-Dateien")
    parser.add_argument("--min_words", type=int, default=100,
                        help="Mindestanzahl an Wörter einer Rede für Topic-Model")
    parser.add_argument("--year", type=int, default=None,
                        help="Jahr der zu downloadenen Plenarreden")
    parser.add_argument("--skip-processing", action="store_true",
                        help="Überspringt alle Verarbeitungsschritte")

    args = parser.parse_args()

    if not args.skip_download:
        data_loader.load_data(args.year)

    if not args.skip_processing:
        print("=" * 60)
        bt_speeches_df = xml_parser.extract_speeches_from_directory("./data/raw/")
        print(f"{len(bt_speeches_df):,} Reden geladen")
        print("=" * 60)
        print("SpaCy Pipeline...")
        nlp = spacy.load("de_core_news_sm")

        text_preprocessing.add_stopwords(
            "data/custom_parliament_stopwords.txt",
            nlp
        )

        bt_speeches_df = text_preprocessing.lemmatize(
            bt_speeches_df,
            "text",
            nlp
        )
        print("=" * 60)
        print("Topic Modeling...")
        df_topic, topic_model = topic_modeling.run_bertopic(
            bt_speeches_df,
            stopwords=nlp.Defaults.stop_words,
            text_col="text",
            min_words=args.min_words,
            use_lemma=True,
        )

        topic_modeling.save_topic_outputs(
            df_topic,
            topic_model,
        )
    else:
        df_topic = pd.read_parquet(
            "data/processed/speeches_with_topics.parquet"
        )
    df_topic = topic_modeling.map_topic_labels(df_topic)
    df_topic_red = topic_modeling.filter_topics(df_topic, 36)
    df_topic_red.to_parquet("data/processed/speeches_with_topics_labeled_red.parquet")


    print("=" * 60)
    print("Pipeline erfolgreich abgeschlossen")
    print("=" * 60)
if __name__ == "__main__":
    main_analysis()   
    
