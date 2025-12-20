LiteFND++

LiteFND++ is a lightweight, interpretable dual-path model for real-time fake news detection in low-resource and CPU-only environments.
It combines TF-IDF features with a Logistic Regression + Naive Bayes ensemble and integrates LIME explanations to provide human-interpretable predictions.

The design prioritizes speed, transparency, and deployability, making LiteFND++ suitable for real-time applications and environments where large transformer models are impractical.

🔍 Key Features

Dual-path ensemble architecture

Logistic Regression (discriminative)

Naive Bayes (generative)

TF-IDF based sparse representations

NER-aware tokenization and semantic normalization

Human-interpretable explanations via LIME

Fast CPU inference (~12 ms per prediction)

No GPU required

Minimal dependencies

Designed for low-resource and real-time settings

📦 Installation

Clone the repository and install the package locally:

git clone https://github.com/toxicovi/LiteFNDpp.git
cd LiteFNDpp
pip install .


Alternatively, install only the required dependencies:

pip install -r requirements.txt


Download the SpaCy English language model:

python -m spacy download en_core_web_sm

🚀 Quick Start
from litefndpp.preprocessing import preprocess_text
from litefndpp.model import LiteFNDpp
from litefndpp.explain import explain_prediction

# Input text
text = "Breaking: miracle cure discovered for disease X!"

# Preprocess
processed_text = preprocess_text(text)

# Initialize and train model
model = LiteFNDpp()
model.fit(train_data, train_labels)

# Predict
prediction = model.predict([processed_text])

# Explain prediction
explanation = explain_prediction(model, processed_text)

print("Prediction:", prediction)
print("Explanation:", explanation)

🧠 Model Overview

LiteFND++ follows a dual-path learning strategy:

Logistic Regression Path

Captures discriminative linear decision boundaries

Robust under sparse TF-IDF representations

Naive Bayes Path

Models word-class generative distributions

Effective in low-data regimes

Predictions from both paths are ensembled to improve robustness while preserving interpretability.

🔎 Explainability

LiteFND++ integrates LIME (Local Interpretable Model-Agnostic Explanations) to provide:

Token-level importance scores

Human-readable explanations

Transparency suitable for journalists, researchers, and policymakers

This aligns with growing concerns about algorithmic accountability in misinformation detection systems.

📊 Dataset

LiteFND++ is evaluated on publicly available fake news datasets, including:

FakeNewsNet

Fake and Real News Dataset (Kaggle)

The framework is dataset-agnostic and can be easily adapted to new corpora.

🧪 Reproducibility

Deterministic preprocessing

CPU-only evaluation

No reliance on stochastic deep models

Lightweight testing via tests/test_model.py

📁 Repository Structure
LiteFNDpp/
├── litefndpp/
│   ├── model.py
│   ├── preprocessing.py
│   ├── explain.py
│   └── utils.py
├── examples/
│   └── demo.ipynb
├── tests/
│   └── test_model.py
├── paper.md
├── paper.bib
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md

📄 Paper

The accompanying research paper is provided in:

paper.md

paper.bib

These files are formatted for JOSS submission.

📜 License

This project is released under the MIT License.
See the LICENSE file for details.

🤝 Citation

If you use LiteFND++ in your research, please cite the accompanying paper.

🌟 Acknowledgments

LIME authors for interpretability tools

Open-source NLP community

Dataset contributors and maintainers

📬 Contact

For questions, issues, or contributions, please open an issue on GitHub.
