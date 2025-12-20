---
title: "LiteFNDpp: A Dual-Path Lightweight and Interpretable Model for Real-Time Fake News Detection"
authors:
  - name: Ovi Pal
    orcid: 0009-0002-6849-162X
    affiliation: 1
affiliations:
  - name: Xi'an Jiaotong University
    index: 1
date: 2025-12-20
bibliography: paper.bib
---

## Summary

The rapid spread of online misinformation poses significant challenges to public trust and information integrity. Recent approaches to fake news detection increasingly rely on transformer-based language models, which achieve strong predictive performance but incur substantial computational costs and offer limited interpretability. These constraints restrict their applicability in real-time and resource-constrained deployment scenarios.

LiteFNDpp is a lightweight and interpretable dual-path ensemble framework designed for efficient fake news detection on CPU-only systems. The model combines term frequency–inverse document frequency (TF–IDF) representations with a calibrated soft-voting ensemble of Logistic Regression and Multinomial Naive Bayes classifiers. Linguistically informed preprocessing, including semantic normalization and named-entity-aware token handling, improves robustness to stylistic and adversarial variations in news content.

To promote transparency, LiteFNDpp integrates Local Interpretable Model-agnostic Explanations (LIME), enabling human-readable explanations for individual predictions. Empirical evaluation demonstrates that the system achieves competitive classification performance while maintaining low inference latency, making it suitable for real-world misinformation monitoring applications where efficiency and interpretability are essential.

---

## Statement of Need

Most state-of-the-art fake news detection systems employ deep neural networks or transformer-based architectures. Although effective, these models typically require specialized hardware, introduce inference latency, and provide limited insight into the rationale behind predictions. Such limitations hinder adoption in domains where explainability, responsiveness, and accessibility are critical.

LiteFNDpp addresses this gap by providing an open-source software package that emphasizes computational efficiency, interpretability, and ease of deployment. The framework is particularly valuable for researchers, journalists, and developers who require a transparent and real-time fake news detection solution without reliance on GPUs or large-scale infrastructure. By combining classical machine learning techniques with modern explainability tools, LiteFNDpp offers a practical alternative to heavyweight deep learning approaches.

**Related Work**  
LiteFND++ is related to previously published JOSS software that emphasizes interpretability and reproducibility, such as `FAT Forensics` [1], `DIANNA` [2], `PyDGN` [3], `TSInterpret` [4], and `imodels` [5]. Unlike these tools, LiteFND++ focuses on **real-time fake news detection** with a **dual-path lightweight ensemble** that achieves CPU-efficient inference with integrated LIME-based explanations.

---

## Implementation

LiteFNDpp is implemented in Python and leverages widely used machine learning and natural language processing libraries. Text preprocessing includes normalization, token filtering, and named-entity-aware transformations prior to feature extraction. Feature representations are constructed using both word-level and character-level TF–IDF vectors to capture lexical and subword patterns.

The classification pipeline consists of a dual-path ensemble combining Logistic Regression and Multinomial Naive Bayes models via weighted soft voting. Model explanations are generated using LIME, enabling localized interpretation of prediction outcomes. The software architecture is modular and extensible, allowing users to adapt preprocessing steps, classifiers, or explanation mechanisms as needed.

---

## Usage Example

```python
from litefndpp.model import LiteFNDpp

# Initialize the model
model = LiteFNDpp()

# Train the model
X_train = ["News article text 1", "News article text 2"]
y_train = [1, 0]  # 1 = Fake, 0 = Real
model.fit(X_train, y_train)

# Predict
prediction = model.predict(["Breaking: miracle cure discovered!"])

# Explain prediction
explanation = model.explain("Breaking: miracle cure discovered!")
print(prediction, explanation)
