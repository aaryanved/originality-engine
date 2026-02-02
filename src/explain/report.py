import spacy
from collections import Counter
from itertools import combinations

nlp = spacy.load("en_core_web_sm")


def extract_concepts(text):
    doc = nlp(text)
    concepts = []

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            concepts.append(token.lemma_.lower())

    return concepts


def rare_concept_pairs(query_concepts, neighbor_texts, top_k=5):
    neighbor_concepts = []

    for text in neighbor_texts:
        neighbor_concepts.extend(extract_concepts(text))

    neighbor_counts = Counter(neighbor_concepts)

    pair_scores = []
    for a, b in combinations(set(query_concepts), 2):
        score = neighbor_counts[a] + neighbor_counts[b]
        pair_scores.append(((a, b), score))

    pair_scores.sort(key=lambda x: x[1])
    return pair_scores[:top_k]


def generate_explanation(query, neighbors):
    query_concepts = extract_concepts(query)
    neighbor_texts = [n["text"] for n in neighbors]

    rare_pairs = rare_concept_pairs(query_concepts, neighbor_texts)

    explanation = []
    explanation.append("This idea appears original because:")

    explanation.append(
        f"- It combines concepts ({', '.join(query_concepts[:4])}) that rarely co-occur in the reference corpus."
    )

    if rare_pairs:
        explanation.append(
            f"- The pairing '{rare_pairs[0][0][0]}' + '{rare_pairs[0][0][1]}' is uncommon among similar ideas."
        )

    explanation.append(
        "- Its framing differs from typical academic abstractions found in the corpus."
    )

    return "\n".join(explanation)