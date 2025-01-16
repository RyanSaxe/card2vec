import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np

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


def check_emb_dir_validity(fnames: list[str]):
    if "embs.npy" not in fnames:
        raise ValueError("Embedding Directory is missing embs.npy")
    if "card_names.npy" not in fnames:
        raise ValueError("Embedding Directory is missing names.npy")
    if "metadata.json" not in fnames:
        raise ValueError("Embedding Directory is missing metadata")


class EmbeddingMetaData(TypedDict):
    model_name: str
    embed_dim: int
    num_cards: int


def load_embeddings(fpath: str | Path, *card_names: str) -> np.typing.NDArray:
    if not isinstance(fpath, Path):
        fpath = Path(fpath)

    files = os.listdir(fpath)
    check_emb_dir_validity(files)

    if len(card_names) == 0:
        ...

    with open(fpath / "metadata.json", "r") as f:
        metadata: EmbeddingMetaData = json.load(f)

    embeddings = np.memmap(
        fpath / "embs.npy", mode="r", dtype=np.float32, shape=(metadata["num_cards"], metadata["embed_dim"])
    )

    if len(card_names) == 0:
        return embeddings

    names = np.load(fpath / "card_names.npy")
    print(names[0])
    indices = [np.where(names == name) for name in card_names]
    print(indices)
    return embeddings[indices]


import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    assert len(args) >= 2
    embeddings = load_embeddings(args[0], *args[1:])
    print(embeddings)
    print(embeddings.shape)
