import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def get_text(
    element: ET.Element | None,
    default: str = "unknown",
) -> str:
    return (
        element.text.strip()
        if element is not None and element.text
        else default
    )

def get_date(root: ET.Element):

    datum = root.find(".//datum")

    if datum is None:
        return None

    return datum.attrib.get("date")

def extract_speaker(rede_xml: ET.Element) -> dict:
    speaker = rede_xml.find(".//redner/name")

    if speaker is None:
        return {
            "first_name": "unknown",
            "last_name": "unknown",
            "fraction": "unknown",
            "role": "unknown",
        }

    return {
        "first_name": get_text(speaker.find("vorname")),
        "last_name": get_text(speaker.find("nachname")),
        "fraction": get_text(speaker.find("fraktion")),
        "role": get_text(speaker.find("rolle/rolle_kurz")),
    }


def extract_speech_content(rede_xml: ET.Element) -> str:
    speech = []

    for paragraph in rede_xml.findall("p"):
        if paragraph.attrib.get("klasse") == "redner":
            continue

        text = " ".join(paragraph.itertext()).strip()

        if text:
            speech.append(text)

    return "\n".join(speech)


def extract_speech(
    rede_xml: ET.Element,
    root: ET.Element,
    filename: str | None = None,
) -> dict:
    speaker = extract_speaker(rede_xml)
    speech = extract_speech_content(rede_xml)
    date = get_date(root)

    return {
        "speech_id": rede_xml.attrib.get("id"),
        "filename": filename,
        "date": date,
        **speaker,
        "text": speech,
        "word_count": len(speech.split()),
        "char_count": len(speech),
    }


def load_xml_file(xml_file: str | Path) -> ET.Element:
    tree = ET.parse(xml_file)
    return tree.getroot()


def extract_speeches_from_file(
    xml_file: str | Path,
) -> list[dict]:
    root = load_xml_file(xml_file)

    speeches = []

    for rede_xml in root.iter("rede"):
        speeches.append(
            extract_speech(
                rede_xml=rede_xml,
                root=root,
                filename=Path(xml_file).name,
            )
        )

    return speeches

def extract_speeches_from_directory(
    xml_dir: str | Path,
) -> pd.DataFrame:
    xml_files = sorted(Path(xml_dir).glob("*.xml"))
    print(f"Processing {len(xml_files)} XML-Files...")

    all_speeches = []

    for file in xml_files:
        all_speeches.extend(extract_speeches_from_file(file))

    df = pd.DataFrame(all_speeches)

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d.%m.%Y",
        errors="coerce",
    )

    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    return df