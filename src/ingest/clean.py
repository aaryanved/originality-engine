import re
from pathlib import Path
from tqdm import tqdm


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) >= 50:
            chunks.append(chunk)

    return chunks


def process_file(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        cleaned = clean_text(line)
        chunks = chunk_text(cleaned)
        all_chunks.extend(chunks)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk + "\n")

    print(f"Saved {len(all_chunks)} chunks.")


if __name__ == "__main__":
    process_file(
        Path("data/raw/arxiv_abstracts.txt"),
        Path("data/processed/arxiv_chunks.txt"),
    )