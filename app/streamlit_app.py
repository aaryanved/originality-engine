import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st

from src.originality.semantic import SemanticOriginality
from src.originality.density import DensityEstimator
from src.originality.score import OriginalityScorer
from src.explain.report import generate_explanation

st.set_page_config(
    page_title="Originality Engine",
    layout="centered",
)

st.title("🧠 Originality Engine")
st.write(
    "Measure how original an idea is relative to a corpus of computer science research abstracts."
)

@st.cache_resource
def load_models():
    semantic = SemanticOriginality()
    density = DensityEstimator()
    scorer = OriginalityScorer()
    baseline = scorer.local_baseline_distribution()
    return semantic, density, scorer, baseline

semantic, density, scorer, baseline = load_models()

idea = st.text_area(
    "Enter your idea:",
    height=120,
    placeholder="e.g. an AI system that evaluates the originality of ideas",
)

if st.button("Evaluate Originality"):
    if not idea.strip():
        st.warning("Please enter an idea.")
    else:
        with st.spinner("Analyzing originality..."):
            semantic_result = semantic.semantic_novelty_score(idea)
            local_density = density.local_density(idea)
            score, percentile = scorer.originality_percentile(idea, baseline)

        st.subheader("📊 Originality Results")
        st.metric("Originality Percentile", f"{percentile:.2f}%")
        st.write(f"**Average semantic distance:** {semantic_result['avg_distance']:.3f}")
        st.write(f"**Closest existing idea distance:** {semantic_result['min_distance']:.3f}")
        st.write(f"**Nearby ideas (local density):** {local_density}")

        st.subheader("🧠 Explanation")
        st.write(generate_explanation(idea, semantic_result["neighbors"]))

        st.subheader("🔍 Nearest Existing Ideas")
        for i, n in enumerate(semantic_result["neighbors"][:3], 1):
            with st.expander(f"Neighbor {i} (distance: {n['distance']:.3f})"):
                st.write(n["text"])