import argparse

from card2vec.cards.embeddings import CardEmbeddings
from card2vec.utils import DATA_DIR

# TODO: let this have command line args for converters and model specifications
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="search embeddings")
    parser.add_argument(
        "--folder_name",
        "-f",
        default="embeddings",
        help=f"{DATA_DIR}folder_name is where the embeddings are",
        type=str,
    )
    parser.add_argument("--card_name", "-c", default="Black Lotus", help="Name of the card to search for", type=str)
    parser.add_argument("--top_n", "-n", default=5, help="Number of closest cards to return", type=int)
    args = parser.parse_args()

    directory = DATA_DIR / args.folder_name

    card_embeddings = CardEmbeddings.load(directory)
    closest_card_embs = card_embeddings.find_closest_n_cards(args.card_name, args.top_n)
    print(list(closest_card_embs.keys()))
