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
````

Download the SpaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

```python
from litefndpp.preprocessing import preprocess_text
from litefndpp.model import LiteFNDpp
from litefndpp.explain import explain_prediction

# Preprocess text
text = "Breaking: miracle cure discovered for disease X!"
processed_text = preprocess_text(text)

# Initialize and train model
model = LiteFNDpp()
model.fit(train_data, train_labels)

# Predict
prediction = model.predict([processed_text])

# Get explanation
explanation = explain_prediction(model, processed_text)
print(prediction, explanation)
```

## Dataset

* [Kaggle FakeNewsNet dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## License

MIT License

```

---

```
