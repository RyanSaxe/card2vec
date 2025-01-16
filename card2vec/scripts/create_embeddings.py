from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from card2vec.cards.load import EmbeddingMetaData, load_cards, simple_converter

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "gpt2"


def create_embeddings(
    card_texts: list[str],
    model_name: str = DEFAULT_MODEL,
    device: Literal["cpu", "cuda"] = DEFAULT_DEVICE,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
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

    return all_embeddings, metadata


# TODO: let this have command line args for converters and model specifications
if __name__ == "__main__":
    card_names, card_texts = zip(*load_cards(simple_converter, max_workers=4))
    embeddings, metadata = create_embeddings(card_texts, batch_size=8)
