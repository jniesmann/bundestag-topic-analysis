import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://search.dip.bundestag.de/api/v1/plenarprotokoll"
API_KEY = os.getenv("BUNDESTAG_API_KEY")
RAW_DATA_DIR = "data/raw"

def request_api():
    """
    Sendet eine GET-Anfrage an die API des Bundestags, um Plenarprotokolle im JSON-Format abzurufen.

    Returns:
        dict: Die JSON-Antwort als dict.
    """
    if API_KEY is None:
        raise ValueError("BUNDESTAG_API_KEY ist nicht gesetzt.")
    
    response = requests.get(url=API_URL, params={
        "format": "json",
        "apikey": API_KEY,
        "f.dokumentart": "plenarprotokoll",
        "f.herausgeber": "BT",
        "f.datum.start": f"2024-01-01",
        "f.datum.end": f"2025-12-31"
    })

    response.raise_for_status()

    return response.json()

def scrape_xml(xml_url):
    """
    Lädt das XML-Dokument herunter und speichert es im Verzeichnis 'data/raw/'.

    Args:
        xml_url (str): Die URL des XML-Dokuments.
    """
    xml = requests.get(url=xml_url, timeout=30)
    xml.raise_for_status()

    filename = xml_url.rsplit("/", 1)[-1]

    os.makedirs(RAW_DATA_DIR, exist_ok=True) 
    full_path = os.path.join(RAW_DATA_DIR, filename)

    with open(full_path, 'w', encoding='utf-8') as output:
        output.write(xml.text)

def load_data():
    """
    Lädt alle Plenarprotokolle des Jahres 2024/25 vom Bundestag herunter
    und gespeichert die zugehörigen XML-Dateien.
    """
    docs = request_api().get('documents', [])

    print(f"Lade {len(docs)} Plenardebatten herunter...")
    for doc in docs:
        if doc.get('herausgeber') == "BT":  # Nur Bundestagsdebatten betrachten
            xml_url = doc.get('fundstelle').get('xml_url')
            scrape_xml(xml_url=xml_url)
    print("Plenarprotokolle erfolgreich heruntergeladen und in data/raw/ gespeichert.")
