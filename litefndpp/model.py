from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class LiteFNDpp:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,4), max_features=10000)
        self.lr = LogisticRegression(C=1.8, max_iter=1000)
        self.nb = MultinomialNB(alpha=0.03)
    
    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.lr.fit(X, labels)
        self.nb.fit(X, labels)
    
    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        p_lr = self.lr.predict_proba(X)
        p_nb = self.nb.predict_proba(X)
        # Weighted soft voting
        return 0.65*p_lr + 0.35*p_nb
    
    def predict(self, texts):
        p = self.predict_proba(texts)
        return np.argmax(p, axis=1)
