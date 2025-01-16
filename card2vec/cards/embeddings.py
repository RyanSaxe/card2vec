import json
import os
import string
import warnings
from pathlib import Path
from typing import Literal, Self, TypedDict

import numpy as np
import torch
from rapidfuzz import fuzz, process
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from card2vec.cards.load import CardConverter, load_cards
from card2vec.utils import DATA_DIR, to_path

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "gpt2"


class EmbeddingMetaData(TypedDict):
    model_name: str
    embed_dim: int
    num_cards: int


class CardEmbeddings:
    def __init__(self, card_names: list[str], embeddings: np.ndarray, metadata: EmbeddingMetaData):
        if len(card_names) != embeddings.shape[0]:
            raise ValueError("Number of card names must match number of embeddings.")

        self.embeddings = embeddings
        self.card_names = card_names
        self.metadata = metadata

    @classmethod
    def create(
        cls,
        converter: CardConverter,
        model_name: str = DEFAULT_MODEL,
        device: Literal["cpu", "cuda"] = DEFAULT_DEVICE,
        batch_size: int = 256,
        max_workers: int = 4,
    ) -> dict[str, np.ndarray]:
        card_names, card_texts = zip(*load_cards(converter, max_workers=max_workers))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO: double check end of string token as padder wont cause model related problems
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        total = len(card_texts)
        metadata = EmbeddingMetaData(model_name=model_name, embed_dim=model.embed_dim, num_cards=total)

        # TODO: possibly should save in chunks? to never have to load it into memory?
        all_embeddings = np.empty(shape=(total, model.embed_dim), dtype=np.float32)
        for start_idx in tqdm(range(0, total, batch_size)):
            end_idx = min(start_idx + batch_size, total)
            batched_card_data = card_texts[start_idx:end_idx]

            # call the transformer with the properly formatted data from the batch
            inputs = tokenizer(batched_card_data, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            # for each text in the batch, extract the corresponding embedding from the transformer
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_length, embed_dim)
            cur_batch_size = last_hidden_state.size(0)
            lengths = inputs["attention_mask"].sum(dim=1).long()
            last_token_idxs = (lengths - 1).clamp(min=0)
            batch_indices = torch.arange(cur_batch_size, device=last_hidden_state.device)
            embeddings = last_hidden_state[batch_indices, last_token_idxs, :]
            all_embeddings[start_idx:end_idx, :] = embeddings.numpy()

        return cls(card_names, all_embeddings, metadata)

    @classmethod
    def load(cls, fpath: str | Path) -> Self:
        fpath = to_path(fpath)

        with open(fpath / "metadata.json", "r") as f:
            metadata: EmbeddingMetaData = json.load(f)

        embeddings = np.memmap(
            fpath / "embs.npy", mode="r", dtype=np.float32, shape=(metadata["num_cards"], metadata["embed_dim"])
        )

        names = np.load(fpath / "card_names.npy")
        return cls(names.tolist(), embeddings, metadata)

    def save(self, dir_name: str | Path | None = None):
        if dir_name is None:
            emb_dir = DATA_DIR / "embs"
        else:
            emb_dir = to_path(dir_name)

        os.makedirs(emb_dir, exist_ok=True)

        with open(emb_dir / "embs.npy", "wb") as f:
            np.save(f, self.embeddings)

        with open(emb_dir / "card_names.npy", "wb") as f:
            np.save(f, self.card_names)

        with open(emb_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def get_embedding(self, card_name: str, fuzzy_search_threshold: float = 0) -> np.ndarray:
        if card_name in self.card_names:
            card_index = np.where(self.card_names == card_name)[0]
            return self.embeddings[card_index]

        if fuzzy_search_threshold == 0:
            raise KeyError(f"{card_name} is not a valid card name.")

        warnings.warn(f"{card_name} is not an exact match. Searching for closest names.")

        def clean_name(name: str) -> str:
            # ensure lowercase, no punctuation, and no trailing whitespace
            name = name.lower().translate(str.maketrans("", "", string.punctuation)).strip()
            # ensure all whitespace is replaced with a single space
            return " ".join(name.split())

        matched_card_name, score, _ = process.extractOne(
            card_name,
            self.card_names,
            scorer=fuzz.ratio,
            processor=clean_name,
        )

        if score >= fuzzy_search_threshold:
            card_index = np.where(self.card_names == matched_card_name)[0]
            return self.embeddings[card_index]

        raise KeyError(
            f"Key '{card_name}''s closest match was {matched_card_name} at a similarity"
            f"of {score}, which is below the required threshold of {fuzzy_search_threshold}"
        )

    def find_closest_n_cards(self, card_name: str, n: int, threshold: float = 80) -> dict[str, np.ndarray]:
        card_embedding = self.get_embedding(card_name, fuzzy_search_threshold=threshold)
        distances = self._compute_all_distances_vectorized(card_embedding)
        closest_indices = np.argsort(distances)
        if closest_indices[0] == card_name:
            warnings.warn(f"Target card {card_name} should always be the closest card. Probably a bug.")
        closest_indices = closest_indices[1 : n + 1]
        closest_card_names = [self.card_names[i] for i in closest_indices]
        closest_embeddings = self.embeddings[closest_indices]
        return dict(zip(closest_card_names, closest_embeddings))

    # NOTE: there aren't enough magic cards to care about doing approximate nearest neighbors, so
    #       we can just do the naive approach where we actually compute the distance to all cards
    def _compute_all_distances_vectorized(self, card_embedding: np.ndarray) -> np.ndarray:
        v = np.squeeze(card_embedding)  # shape (embedding_dim,)
        norm_v = np.linalg.norm(v)
        norms = np.linalg.norm(self.embeddings, axis=1)
        # calculate cosine distance
        return 1 - self.embeddings.dot(v) / (norms * norm_v)


if __name__ == "__main__":
    card_embeddings = CardEmbeddings.load(DATA_DIR / "embs")
    closest_cards = card_embeddings.find_closest_n_cards("Elvish Mystic", 5)
    print(closest_cards.keys())
