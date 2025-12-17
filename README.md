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

Copyright (c) 2025 Ovi Pal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

---

```
