import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

class Preprocessor:
    """
    Preprocessing pipeline for LiteFND++:
    - NER-aware token joining
    - Semantic normalization
    - TF-IDF feature extraction
    """
    def __init__(self, max_features: int = 10000, ngram_range=(1, 4), char_ngram_range=(3, 6)):
        self.tfidf = TfidfVectorizer(
            tokenizer=self._tokenize,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=char_ngram_range,
            max_features=max_features
        )
        
    def _tokenize(self, text: str) -> List[str]:
        """
        NER-aware tokenization with semantic normalization:
        - Joins named entities as single tokens (e.g., Barack_Obama)
        - Replaces numbers and dates with placeholders
        - Removes stopwords and punctuation
        """
        doc = nlp(text)
        tokens = []
        
        # Named Entity Joining
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                tokens.append(ent.text.replace(' ', '_'))
                
        # Token processing
        for token in doc:
            if token.like_num:
                tokens.append("[NUM]")
            elif token.is_alpha and not token.is_stop:
                tokens.append(token.lemma_.lower())
        
        return tokens

    def fit_transform(self, texts: List[str]):
        """
        Fits the TF-IDF vectorizers and transforms the text into feature matrices
        """
        word_features = self.tfidf.fit_transform(texts)
        char_features = self.char_tfidf.fit_transform(texts)
        
        # Combine word + character TF-IDF
        import numpy as np
        return np.hstack([word_features.toarray(), char_features.toarray()])

    def transform(self, texts: List[str]):
        """
        Transforms new text using the fitted vectorizers
        """
        word_features = self.tfidf.transform(texts)
        char_features = self.char_tfidf.transform(texts)
        import numpy as np
        return np.hstack([word_features.toarray(), char_features.toarray()])
