import argparse
from functools import partial

from card2vec.cards.embeddings import DEFAULT_MODEL, CardEmbeddings
from card2vec.cards.load import card_to_prompt, load_cards
from card2vec.utils import DATA_DIR

# TODO: let this have command line args for converters and model specifications
if __name__ == "__main__":
    card_names, card_texts = zip(*load_cards(card_to_prompt, max_workers=4))
    parser = argparse.ArgumentParser(description="Process folder and model parameters")
    parser.add_argument("--folder_name", "-f", default="embeddings", help=f"create {DATA_DIR}folder_name", type=str)
    parser.add_argument(
        "--model_name", "-m", default=DEFAULT_MODEL, help="Name of the huggingface model to use", type=str
    )
    parser.add_argument("--batch_size", "-b", default=8, help="Batch size for inference", type=int)

    args = parser.parse_args()

    directory = DATA_DIR / args.folder_name

    converter = partial(
        card_to_prompt,
        card_properties=["color_identity", "oracle_text", "power", "toughness", "loyalty"],
    )

    card_embeddings = CardEmbeddings.create(converter=converter, batch_size=args.batch_size, model_name=args.model)
    card_embeddings.save(directory)
