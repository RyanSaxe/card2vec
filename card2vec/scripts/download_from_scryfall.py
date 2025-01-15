import os

import requests
from tqdm import tqdm

from card2vec.utils import DATA_DIR


def download_card_data() -> None:
    url = "https://api.scryfall.com/bulk-data/oracle-cards"
    download_link = requests.get(url).json()["download_uri"]

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "cards.json")

    with requests.get(download_link, stream=True) as response:
        response.raise_for_status()

        with open(file_path, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                file.write(chunk)


if __name__ == "__main__":
    download_card_data()
