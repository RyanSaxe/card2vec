from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from card2vec.cards.text import load_cards, simple_converter


def compute_paragraph_vectors_batch(
    card_names,
    card_texts,
    model_name: str = "gpt2",
    device: str = "cpu",
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Computes paragraph vectors for a dictionary of titles and paragraphs using batching.

    Args:
        paragraph_dict (Dict[str, str]): Dictionary mapping titles to paragraphs.
        model_name (str): Hugging Face model identifier.
        device (str): Device to run the model on ('cpu' or 'cuda').
        batch_size (int): Number of paragraphs to process in each batch.
        max_length (int): Maximum token length for each paragraph.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping titles to their corresponding paragraph vectors.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Use the EOS token as the padding token
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    total = len(card_texts)
    title_to_vector = {}

    # Process in batches
    for start_idx in tqdm(range(0, total, batch_size)):
        end_idx = min(start_idx + batch_size, total)
        batch_titles = card_names[start_idx:end_idx]
        batch_paragraphs = card_texts[start_idx:end_idx]

        # Tokenize the batch with padding and truncation
        inputs = tokenizer(batch_paragraphs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model outputs without computing gradients
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the last hidden states
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)

        # For each example in the batch, get the last non-padded token's hidden state
        for i, title in enumerate(batch_titles):
            # Find the actual length (number of tokens) for the i-th example
            # Assuming attention_mask is 1 for real tokens and 0 for padding
            attention_mask = inputs["attention_mask"][i]
            # Get indices where attention_mask is 1
            non_padded_indices = torch.nonzero(attention_mask, as_tuple=True)[0]
            if len(non_padded_indices) == 0:
                breakpoint()
            last_token_idx = non_padded_indices[-1].item()
            paragraph_vector = last_hidden_state[i, last_token_idx, :].cpu().numpy()
            title_to_vector[title] = paragraph_vector

    return title_to_vector


# Example usage:
if __name__ == "__main__":
    card_names, card_texts = zip(*load_cards(simple_converter, max_workers=4))

    vectors = compute_paragraph_vectors_batch(
        card_names,
        card_texts,
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=4,  # Adjust based on your hardware
    )

    for title, vec in vectors.items():
        print(f"{title}: vector shape = {vec.shape}")
        print(vec)
        break
