from src.explain.report import generate_explanation
from src.originality.semantic import SemanticOriginality
from src.originality.density import DensityEstimator
from src.originality.score import OriginalityScorer


if __name__ == "__main__":
    idea = "an ai system that evaluates originality of ideas"

    semantic = SemanticOriginality()
    density = DensityEstimator()
    scorer = OriginalityScorer()

    semantic_result = semantic.semantic_novelty_score(idea)
    local_density = density.local_density(idea)
    baseline = scorer.local_baseline_distribution()
    score, percentile = scorer.originality_percentile(idea, baseline)

    print("\n=== SEMANTIC NOVELTY ===")
    print("Average distance:", semantic_result["avg_distance"])
    print("Closest distance:", semantic_result["min_distance"])

    print("\n=== LOCAL DENSITY ===")
    print("Nearby ideas within radius:", local_density)

    print("\n=== ORIGINALITY SCORE ===")
    print("Mean distance:", score)
    print(f"Originality percentile: {percentile:.2f}%")

    print("\n=== NEAREST IDEAS ===")
    for n in semantic_result["neighbors"][:3]:
        print("-", n["text"][:120], "...")

    print("\n=== EXPLANATION ===")
    print(generate_explanation(idea, semantic_result["neighbors"]))