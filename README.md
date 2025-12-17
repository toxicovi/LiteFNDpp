# LiteFNDpp

**LiteFND++** is a lightweight and interpretable dual-path model for real-time fake news detection.  
It combines TF-IDF features with a Logistic Regression + Naive Bayes ensemble and integrates LIME explanations for transparency.

## Features

- Dual-path ensemble of Logistic Regression and Naive Bayes
- NER-aware tokenization and semantic normalization
- Fast inference on CPU (~12ms per prediction)
- Human-interpretable explanations via LIME
- Suitable for real-time, low-resource environments

## Installation

Clone the repository or download the ZIP, then install dependencies:

```bash
pip install -r requirements.txt
