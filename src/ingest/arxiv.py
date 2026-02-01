import requests
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def fetch_arxiv_abstracts(
    search_query="cat:cs.*",
    max_results=500,
    batch_size=100,
    sleep_time=3,
):
    abstracts = []

    for start in tqdm(range(0, max_results, batch_size)):
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": batch_size,
        }

        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()

            abstracts.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "source": "arxiv",
                }
            )

        time.sleep(sleep_time)

    return abstracts


def save_raw_abstracts(abstracts, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in abstracts:
            f.write(item["abstract"].replace("\n", " ") + "\n")


if __name__ == "__main__":
    data = fetch_arxiv_abstracts(max_results=500)
    save_raw_abstracts(data, Path("data/raw/arxiv_abstracts.txt"))
    print(f"Saved {len(data)} abstracts.")