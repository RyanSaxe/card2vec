import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypedDict

from card2vec.utils import DATA_DIR


# TODO: create types that contains all scryfall valid card fields
class Card(TypedDict, total=False):
    name: str
    oracle_text: str
    card_faces: list[dict[str, str]]
    ...


CardConverter = Callable[[Card], tuple[str, str]]


def load_cards(converter: CardConverter, max_workers: int = 4) -> list[tuple[str, str]]:
    with open(DATA_DIR / "cards.json", "r") as file:
        cards = json.load(file)

    # be able to skip stuff like tokens/emblems/etc
    # ensure names card["name"] is unique

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


def to_xml(card: Card, outer_tag: str | None, tags: list[str]):
    tagged = [f"<{tag}>{card[tag]}</{tag}>" for tag in tags if tag in card and card[tag] not in [None, ""]]
    if outer_tag is not None:
        tagged = [f"<{outer_tag}>", *tagged, f"</{outer_tag}>"]
    return "\n".join(tagged)


def card_to_prompt(card: Card, card_properties: list[str]) -> tuple[str, str]:
    if "card_faces" in card:
        front_xml = to_xml(card["card_faces"][0], outer_tag="front", tags=card_properties)
        back_xml = to_xml(card["card_faces"][1], outer_tag="back", tags=card_properties)
        return card["name"], f"<card>\n{front_xml}\n{back_xml}\n</card>"
    return card["name"], to_xml(card, outer_tag="card", tags=card_properties)
