# Originality Engine V1.0.0
##### By Aaryan Ved Bhalla

Originality Engine is a research NLP project that explores how conceptual originality can be measured relative to existing knowledge distributions.

The system evaluates ideas by comparing them against a reference corpus using semantic searching and local density analysis, with an emphasis on evidence based results.

## Overview

- Computes relative originality scores
- Uses semantic searching and nearest neighbor retrieval
- Provides transparent metrics and comparisons
- Designed for research and exploratory analysis

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/originality-engine.git
cd originality-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py