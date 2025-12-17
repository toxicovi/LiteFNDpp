---
title: "LiteFNDpp: A Dual-Path Lightweight and Interpretable Model for Real-Time Fake News Detection"
authors:
  - name: Ovi Pal
    orcid: 0009-0002-6849-162X
    affiliation: "Xi'An Jiaotong University"
    email: sendmail2ovi@gmail.com
---


# Summary

The proliferation of digital misinformation poses significant challenges to information integrity worldwide. While transformer-based models have demonstrated impressive performance in fake news detection, their substantial computational requirements and limited interpretability hinder practical deployment. 

**LiteFNDpp** is a novel dual-path ensemble framework that achieves an optimal balance between performance, efficiency, and explainability. It leverages sophisticated linguistic preprocessing—including semantic normalization and NER-aware token joining—combined with comprehensive TF-IDF vectorization using character and word n-grams. The model integrates Logistic Regression and Multinomial Naive Bayes through calibrated soft-voting, achieving state-of-the-art performance with an F1-score of 0.991 on the Kaggle FakeNewsNet dataset, while maintaining robust performance (F1=0.75) on the challenging LIAR dataset. 

A key innovation is the native integration of Local Interpretable Model-agnostic Explanations (LIME), providing transparent, human-interpretable rationales for classification decisions. LiteFNDpp achieves inference times of 12ms on CPU-only systems, representing an 18-20x speedup over BERT-based models while maintaining competitive accuracy. Extensive experiments demonstrate that LiteFNDpp outperforms existing approaches across critical dimensions of accuracy, computational efficiency, and explainability, offering a practical and deployable solution for real-world misinformation detection scenarios where resource constraints and decision transparency are paramount.

# Statement of Need

Current state-of-the-art models for fake news detection rely heavily on transformer-based architectures. Although accurate, these models are computationally expensive, require GPUs, and provide limited interpretability. 

LiteFNDpp addresses these limitations by offering:

- **High efficiency**: CPU-friendly inference at 12ms per sample.
- **Transparency**: LIME-based explanations for all predictions.
- **Robust performance**: Comparable or superior F1-scores to transformer models on multiple datasets.
- **Practical deployment**: Lightweight architecture suitable for mobile, edge, or low-resource environments.

This makes LiteFNDpp particularly useful for researchers, journalists, and developers who need a reliable, interpretable, and real-time fake news detection system without relying on specialized hardware.

# Implementation

LiteFNDpp is implemented in Python 3.10+ and leverages standard ML and NLP libraries.

## Installation

```bash
git clone https://github.com/toxicovi/LiteFNDpp.git
cd LiteFNDpp
pip install -r requirements.txt
````

## Usage Example

```python
from litefndpp.model import LiteFNDpp

# Initialize model
model = LiteFNDpp()

# Training
X_train = ["News article 1 text", "News article 2 text", ...]
y_train = [1, 0, ...]  # 1 = Fake, 0 = Real
model.fit(X_train, y_train)

# Prediction
X_test = ["Test article text"]
predictions = model.predict(X_test)
proba = model.predict_proba(X_test)

# Explanation
exp = model.explain("Test article text")
print(exp)
```

## Folder Structure

```
LiteFNDpp/
│
├── README.md
├── LICENSE
├── requirements.txt
├── litefndpp/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── explain.py
│   └── utils.py
├── examples/
│   └── demo.ipynb
└── tests/
    └── test_model.py
```

# Functional Description

LiteFNDpp combines:

* **Dual-path ensemble**: Logistic Regression + Multinomial Naive Bayes with weighted soft-voting.
* **Advanced preprocessing**: NER-aware tokenization, semantic normalization, TF-IDF vectorization.
* **Cognitive load analysis**: Detects manipulative language patterns.
* **Adversarial augmentation**: Generates realistic fake news variants to improve robustness.
* **Cross-platform consistency analysis**: Checks semantic similarity across sources.
* **Explainability**: LIME-based local explanations for all predictions.

# Quality Control

* Evaluated on **Kaggle FakeNewsNet** and **LIAR** datasets.
* Inference time benchmarked on CPU-only systems.
* Ablation studies demonstrate the impact of each component (dual-path ensemble, NER-aware tokenization, semantic normalization).
* Unit tests available in `tests/test_model.py`.

# (Optional) Example Output

```python
exp = model.explain("Breaking: new miracle cure discovered!")
print(exp)
```

Output:

```text
{
  "top_features": [("miracle_cure", 0.42), ("breaking", 0.35), ...],
  "cognitive_scores": {"emotional": 0.8, "logical_gaps": 0.3, ...},
  "prediction": 1
}
```

# References

# References

1. H. Allcott and M. Gentzkow, “Social media and fake news in the 2016 election,” *Journal of Economic Perspectives*, vol. 31, no. 2, pp. 211–236, 2017.  
2. S. Vosoughi, D. Roy, and S. Aral, “The spread of true and false news online,” *Science*, vol. 359, no. 6380, pp. 1146–1151, 2018.  
3. K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, “Fake news detection on social media: A data mining perspective,” *ACM SIGKDD Explorations*, vol. 19, no. 1, pp. 22–36, 2017.  
4. J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in *NAACL*, 2019.  
5. Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv preprint arXiv:1907.11692, 2019.  
6. T. B. Brown et al., “Language models are few-shot learners,” in *NeurIPS*, 2020.  
7. Z. C. Lipton, “The mythos of model interpretability,” *Communications of the ACM*, vol. 61, no. 10, pp. 36–43, 2018.  
8. M. T. Ribeiro, S. Singh, and C. Guestrin, “Why should I trust you?: Explaining the predictions of any classifier,” in *KDD*, 2016, pp. 1135–1144.  
9. C. Castillo, M. Mendoza, and B. Poblete, “Information credibility on Twitter,” in *WWW*, 2011, pp. 675–684.  
10. N. Ruchansky, S. Seo, and Y. Liu, “CSI: A hybrid deep model for fake news detection,” in *CIKM*, 2017, pp. 797–806.  
11. V. Pérez-Rosas, B. Kleinberg, A. Lefevre, and R. Mihalcea, “Automatic detection of fake news,” in *COLING*, 2018, pp. 3391–3401.  
12. Y. Wang et al., “EANN: Event adversarial neural networks for multi-modal fake news detection,” in *KDD*, 2018, pp. 849–857.  
13. Y. Long, Q. Lu, R. Xiang, and M. Li, “Fake news detection through NLP,” *IEEE Access*, 2017.  
14. Y. Belinkov and J. Glass, “Analysis methods in neural language processing: A survey,” *Transactions of the Association for Computational Linguistics*, vol. 7, pp. 49–72, 2019.  
15. X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for text classification,” in *NIPS*, 2015, pp. 649–657.  
16. T. Joachims, “Text categorization with support vector machines: Learning with many relevant features,” in *ECML*, 1998, pp. 137–142.  
17. J. D. M. Rennie et al., “Tackling the poor assumptions of naive Bayes text classifiers,” in *ICML*, 2003, pp. 616–623.  
18. K. Shu, D. Mahudeswaran, and H. Liu, “FakeNewsNet: A data repository with news content, social context, and dynamic information,” *Big Data*, 2020.  
19. S. Kumar and K. M. Carley, “Tree LSTM with hierarchical attention for fake news detection,” in *ACL*, 2019.  
20. Clément Bisaillon, “Fake and Real News Dataset,” Kaggle, https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset.  
21. M. Honnibal and I. Montani, “spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing,” 2017. [Online]. Available: https://spacy.io  
22. S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*, O'Reilly Media, 2009.  
23. C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.  
24. A. Y. Ng, “Feature selection, L1 vs. L2 regularization, and rotational invariance,” in *ICML*, 2004, pp. 78–85.  
25. A. McCallum and K. Nigam, “A comparison of event models for naive Bayes text classification,” in *AAAI Workshop on Learning for Text Categorization*, 1998, pp. 41–48.  
26. R. Binns et al., “It's reducing a human being to a percentage: Citizens' concerns over algorithmic decision-making,” in *CHI*, 2018, pp. 1–14.  
27. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” arXiv preprint arXiv:1910.01108, 2019.  

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

---

