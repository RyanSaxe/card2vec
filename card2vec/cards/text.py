import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypedDict

from card2vec.utils import DATA_DIR


class Card(TypedDict, total=False):
    name: str
    oracle_text: str
    card_faces: list[dict[str, str]]
    ...


CardConverter = Callable[[Card], tuple[str, str]]


def load_cards(converter: CardConverter, max_workers: int = 4) -> list[tuple[str, str]]:
    with open(DATA_DIR / "cards.json", "r") as file:
        cards = json.load(file)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(converter, cards))


def simple_converter(card: Card) -> tuple[str, str]:
    if "card_faces" in card:
        oracle_text = "\n---\n".join(face["oracle_text"] for face in card["card_faces"])
    else:
        oracle_text = card["oracle_text"]
    if len(oracle_text.strip()) == 0:
        oracle_text = "VANILLA"
    return card["name"], oracle_text
