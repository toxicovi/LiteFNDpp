---
title: "LiteFNDpp: A Dual-Path Lightweight and Interpretable Model for Real-Time Fake News Detection"
authors:
  - name: Ovi Pal
    orcid: 0000-0000-0000-0000
    affiliation: "Independent Researcher"
    email: ovi@example.com
---

# Summary

The proliferation of digital misinformation poses significant challenges to information integrity worldwide. LiteFNDpp is a lightweight and interpretable Python package for real-time fake news detection. It implements a dual-path ensemble of **Logistic Regression** and **Multinomial Naive Bayes**, combined with sophisticated text preprocessing (NER-aware tokenization and semantic normalization) and TF-IDF feature extraction. LiteFNDpp also integrates **LIME** for transparent explanations of model predictions.

The package achieves high accuracy while maintaining minimal computational requirements, enabling deployment in resource-constrained environments such as personal computers, edge devices, and mobile platforms.

# Statement of Need

Modern fake news detection often relies on deep learning and transformer-based models, which require significant GPU resources and are opaque in their decision-making. This makes them unsuitable for real-time deployment or for applications requiring explainable outputs. LiteFNDpp addresses these limitations by:

- Offering **CPU-friendly, low-latency inference** suitable for real-time use.
- Providing **interpretable predictions** via LIME, allowing end-users to understand the rationale behind classifications.
- Maintaining **state-of-the-art performance** on benchmark datasets without the computational overhead of large neural networks.

# Key Features

- **Dual-path ensemble:** Combines Logistic Regression and Multinomial Naive Bayes with calibrated soft-voting.
- **Advanced preprocessing:** NER-aware tokenization, semantic normalization of dates, numbers, and locations, and TF-IDF vectorization with word and character n-grams.
- **Explainability:** Local Interpretable Model-agnostic Explanations (LIME) integrated for human-readable classification rationales.
- **Adversarial robustness:** Optional adversarial data augmentation to simulate real-world misinformation variants.
- **Cross-platform consistency checks:** Detects story variants across multiple sources.
- **Lightweight:** Minimal memory footprint (~16 MB) and fast CPU inference (~12 ms per sample).

# Example Usage

```python
from litefndpp.model import LiteFNDpp

# Sample data
texts = [
    "Breaking: Scientists claim a miracle cure for COVID-19!",
    "The government announced new policies to improve healthcare."
]

# Initialize and train the model
model = LiteFNDpp()
# X_train, y_train = ...  # Load your dataset
# model.fit(X_train, y_train)

# Predict probabilities
proba = model.predict_proba(texts)
print(proba)

# Predict labels
labels = model.predict(texts)
print(labels)

# Explain a single prediction
explanation = model.explain(texts[0])
print(explanation)
