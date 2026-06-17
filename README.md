# Bundestag Topic Explorer

Interaktive Analyse von Bundestagsreden mittels NLP und Topic Modeling.

Das Projekt verarbeitet die offiziellen XML-Protokolle der Plenardebatten des Deutschen Bundestages, extrahiert Redebeiträge, identifiziert thematische Schwerpunkte mit BERTopic und stellt die Ergebnisse in einem interaktiven Streamlit-Dashboard dar.

## Live Demo

http://141.5.107.20:8501

## Dashboard

Das Dashboard ermöglicht die Exploration politischer Debatten nach Themenclustern.

Für jedes identifizierte Thema werden dargestellt:

- WordCloud der häufigsten Begriffe
- Zeitliche Entwicklung des Themas
- Fraktionsspezifische Themenschwerpunkte
- Repräsentative Beispielreden

### Themenübersicht

![Dashboard Overview](screenshots/dashboard_overview.png)

### Beispiel: WordCloud aller Reden zum Thema Migration & Asylpolitik

![Migration Topic](screenshots/migration_topic.png)

### Beispiel: Plenardebatten zum Thema Schwangerschaftsabbruch im Zeitverlauf

![Ukraine Topic](screenshots/sa_topic.png)

## Datenquelle

Die Daten stammen aus den öffentlich verfügbaren XML-Protokollen des Deutschen Bundestages:

https://www.bundestag.de/services/opendata

Berücksichtigt wurden Plenardebatten der Jahre 2024 und 2025.

## Methodik

### 1. Datenextraktion

- Download der XML-Protokolle
- Extraktion einzelner Redebeiträge
- Erfassung von Sprecherinformationen, Fraktion und Datum

### 2. Textvorverarbeitung

- spaCy (`de_core_news_sm`)
- Lemmatisierung
- Entfernung deutscher Stopwörter
- Entfernung bundestagsspezifischer Stopwörter

### 3. Topic Modeling

- SentenceTransformer Embeddings
- BERTopic
- Manuelle Benennung der Themencluster anhand von Schlüsselbegriffen und repräsentativen Reden

### 4. Dashboard

- Streamlit
- Matplotlib
- WordCloud

## Projektstruktur

## Projektstruktur

```text
bundestag-topic-analysis/
│
├── bt_pipeline/
│   ├── data_loader.py          # Download der Bundestagsdaten
│   ├── xml_parser.py           # Extraktion der Redebeiträge aus XML
│   ├── text_preprocessing.py   # spaCy-Vorverarbeitung & Lemmatisierung
│   ├── topic_modeling.py       # BERTopic-Modellierung
│   └── topic_labels.py         # Manuelle Themenzuordnung
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard.py                # Streamlit Dashboard
├── main.py                     # Ausführung der Datenpipeline
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## Lokale Ausführung

### Dashboard

Das Dashboard kann ohne vorherige Ausführung der Datenpipeline gestartet werden, da alle für die Visualisierung benötigten Dateien bereits im Repository enthalten sind.

```bash
streamlit run dashboard.py
```

### Pipeline zum Datapreprocessing

Die Datenpipeline umfasst den Download der Bundestagsprotokolle, die Vorverarbeitung der Redebeiträge sowie das Topic Modeling mittels BERTopic. Sie kann über `main.py` ausgeführt werden.
Da BERTopic auf stochastischen Komponenten basiert, können bei einer erneuten Ausführung geringfügige Abweichungen in den resultierenden Themenclustern und deren Zuordnung auftreten.

#### Vollständige Pipeline

```bash
python main.py
```

Für den Download der Bundestagsdaten wird eine `.env`-Datei verwendet.
Beispiel:

```env
API_KEY=key

Der aktuelle API-Key kann unter https://dip.bundestag.de/%C3%BCber-dip/hilfe/api eingesehen werden.

#### Download überspringen

```bash
python main.py --skip-download
```

#### Vorverarbeitung und Topic Modeling überspringen

```bash
python main.py --skip-processing
```


## Docker

Das Dashboard kann containerisiert betrieben werden:

```bash
docker compose up -d --build
```

Anschließend ist das Dashboard unter folgendem Endpunkt erreichbar:

http://localhost:8501

## Verwendete Technologien

- Python
- pandas
- spaCy
- BERTopic
- SentenceTransformers
- Streamlit
- Docker / Podman

## Autor

Jan Niesmann

GitHub: https://github.com/jniesmann
