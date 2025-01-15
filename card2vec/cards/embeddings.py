from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer

Device = Literal["cpu", "cuda"]

CardData: list[str]


def build_card_vectors(cards: CardData, model_name: str = "gpt2", device: Device = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    title_to_vector = {}
    for title, paragraph in paragraph_dict.items():
        inputs = tokenizer(paragraph, return_tensors="pt", truncation=True)

        # Get model outputs without computing gradients
        with torch.no_grad():
            outputs = model(inputs)

        # Extract the last hidden state: shape (batch_size, sequence_length, hidden_dim)
        last_hidden_state = outputs.last_hidden_state

        # Select the embedding for the last token of the sequence
        # [0, -1, :] picks the first (and only) item in the batch, last token, all features
        paragraph_vector = last_hidden_state[0, -1, :].cpu().numpy()

        # Map the title to its corresponding paragraph vector
        title_to_vector[title] = paragraph_vector

    return title_to_vector


# Example usage:
paragraphs = {
    "Title1": "This is the first paragraph. It contains multiple sentences.",
    "Title2": "Here is another paragraph. It also has several sentences.",
}

vectors = compute_paragraph_vectors(paragraphs)
for title, vec in vectors.items():
    print(f"{title}: vector shape = {vec.shape}")
