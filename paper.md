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

The rapid spread of online misinformation poses significant challenges to public trust and information integrity. Recent approaches to fake news detection increasingly rely on transformer-based language models, which achieve strong predictive performance but incur substantial computational costs and offer limited interpretability [@devlin2019bert; @liu2019roberta; @brown2020gpt3]. These constraints restrict their applicability in real-time and resource-constrained deployment scenarios.

LiteFNDpp is a lightweight and interpretable dual-path ensemble framework designed for efficient fake news detection on CPU-only systems. The model combines term frequency–inverse document frequency (TF–IDF) representations with a calibrated soft-voting ensemble of Logistic Regression and Multinomial Naive Bayes classifiers [@joachims1998svm; @rennie2003nb]. Linguistically informed preprocessing, including semantic normalization and named-entity-aware token handling [@honnibal2017spacy; @bird2009nltk], improves robustness to stylistic and adversarial variations in news content.

To promote transparency, LiteFNDpp integrates Local Interpretable Model-agnostic Explanations (LIME) [@ribeiro2016lime], enabling human-readable explanations for individual predictions. Empirical evaluation demonstrates that the system achieves competitive classification performance while maintaining low inference latency, making it suitable for real-world misinformation monitoring applications where efficiency and interpretability are essential.

---

## Statement of Need

Most state-of-the-art fake news detection systems employ deep neural networks or transformer-based architectures. Although effective, these models typically require specialized hardware, introduce inference latency, and provide limited insight into the rationale behind predictions [@lipton2018mythos]. Such limitations hinder adoption in domains where explainability, responsiveness, and accessibility are critical.

LiteFNDpp addresses this gap by providing an open-source software package that emphasizes computational efficiency, interpretability, and ease of deployment. The framework is particularly valuable for researchers, journalists, and developers who require a transparent and real-time fake news detection solution without reliance on GPUs or large-scale infrastructure. By combining classical machine learning techniques with modern explainability tools, LiteFNDpp offers a practical alternative to heavyweight deep learning approaches.

**Related Work**  
LiteFNDpp builds on prior interpretability-focused software such as `FAT Forensics` and ensemble-based NLP classifiers [@shu2017fakenews; @castillo2011credibility; @ruchansky2017csi]. Unlike these tools, LiteFNDpp focuses on **real-time fake news detection** with a **dual-path lightweight ensemble** that achieves CPU-efficient inference with integrated LIME-based explanations.

---

## Implementation

LiteFNDpp is implemented in Python and leverages widely used NLP libraries [@honnibal2017spacy; @bird2009nltk] and machine learning tools. Text preprocessing includes:

- Lowercasing, punctuation removal, and token filtering
- Named-entity-aware transformations (e.g., "Barack Obama" → "Barack_Obama")
- Semantic normalization of dates, numerals, and locations

Feature representations are constructed using **word-level and character-level TF–IDF vectors** [@manning2008ir; @zhang2015charcnn] to capture lexical and subword patterns.

The classification pipeline consists of a **dual-path ensemble**:

- **Logistic Regression (LR)** – handles high-dimensional TF–IDF features [@joachims1998svm]  
- **Multinomial Naive Bayes (NB)** – models term-frequency distributions [@mccallum1998event; @rennie2003nb]  

Predictions are combined via **weighted soft voting**:

$$
P_{\text{final}}(y) = 0.65 \cdot P_{\text{LR}}(y) + 0.35 \cdot P_{\text{NB}}(y)
$$

Explanations are generated using **LIME**, producing localized and human-readable interpretations [@ribeiro2016lime].

The software architecture is **modular and extensible**, allowing users to modify preprocessing, classifiers, or explanation mechanisms.

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
````

---

## Features

* **Dual-Path Ensemble:** Logistic Regression + Naive Bayes via weighted soft voting
* **CPU-Efficient:** ~16 MB model size, inference ~12 ms
* **Linguistically Informed Preprocessing:** NER-aware token joining, semantic normalization
* **TF–IDF Feature Engineering:** Word n-grams (1–4) and character n-grams (3–6)
* **Explainability:** LIME provides token-level insights into predictions
* **Modular Architecture:** Customizable preprocessing, models, and explainability tools

---

## Performance

| Model                            | Accuracy | F1-Score | Inference Time (ms) | Explainable |
| -------------------------------- | -------- | -------- | ------------------- | ----------- |
| BERT-base [@devlin2019bert]      | 97.8     | 0.979    | 243                 | No          |
| RoBERTa-base [@liu2019roberta]   | 98.2     | 0.981    | 221                 | No          |
| DistilBERT [@sanh2019distilbert] | 97.6     | 0.976    | 112                 | No          |
| TF–IDF + LR                      | 98.4     | 0.983    | 17                  | Yes         |
| **LiteFNDpp**                    | 99.0     | 0.991    | 12                  | Yes         |

* LiteFNDpp achieves **state-of-the-art performance** while being **18–20× faster than transformers** on CPU.
* Ablation studies confirm the contribution of dual-path ensemble, NER token joining, and semantic normalization.

---

## Evaluation

* **Cross-Dataset Generalization:** Trained on FakeNewsNet [@shu2020fakenews], tested on LIAR; LiteFNDpp maintains performance advantage.
* **Explainability Analysis:** LIME highlights words supporting "Fake" or "Real" predictions [@ribeiro2016lime].
* **Computational Efficiency:** Model size ~16 MB; training completes in ~5 minutes on CPU.

**Ablation Study Example (F1-score)**

| Configuration                             | F1-Score |
| ----------------------------------------- | -------- |
| TF–IDF + LR (baseline)                    | 0.983    |
| + Naive Bayes Ensemble                    | 0.987    |
| + NER-aware Token Joining                 | 0.989    |
| + Semantic Normalization (Full LiteFNDpp) | 0.991    |

---

## How to Cite

```bibtex
@article{LiteFNDpp2025,
  title = {LiteFNDpp: A Dual-Path Lightweight and Interpretable Model for Real-Time Fake News Detection},
  author = {Ovi Pal},
  journal = {Journal of Open Source Software},
  year = {2025},
  volume = {10},
  number = {xx},
  pages = {xxxx},
  doi = {10.21105/joss.xxxxx}
}
```

```

