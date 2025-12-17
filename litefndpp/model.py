from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class LiteFNDppModel:
    """
    Basic LiteFND++ ensemble: Logistic Regression + Naive Bayes
    """
    def __init__(self):
        self.lr = LogisticRegression(C=1.8, max_iter=1000)
        self.nb = MultinomialNB(alpha=0.03)
        self.lr_weight = 0.65
        self.nb_weight = 0.35
        
    def fit(self, X, y):
        self.lr.fit(X, y)
        self.nb.fit(X, y)
        
    def predict_proba(self, X):
        lr_proba = self.lr.predict_proba(X)
        nb_proba = self.nb.predict_proba(X)
        return self.lr_weight * lr_proba + self.nb_weight * nb_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:,1] > 0.5).astype(int)

