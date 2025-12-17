from lime.lime_text import LimeTextExplainer
from typing import List, Dict

class Explainer:
    """
    LIME explainer for LiteFND++ predictions
    """
    def __init__(self, class_names: List[str] = ['Real', 'Fake'], kernel_width: int = 25):
        self.explainer = LimeTextExplainer(
            class_names=class_names,
            kernel_width=kernel_width,
            verbose=False
        )

    def explain_instance(self, text: str, predict_fn, num_features: int = 5, num_samples: int = 500) -> Dict:
        """
        Generate explanation for a single text instance
        Args:
            text: Input text
            predict_fn: Prediction function returning probabilities
            num_features: Top features to display
            num_samples: LIME sampling size
        Returns:
            Dictionary containing top features and their weights
        """
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        return {
            "top_features": exp.as_list()
        }
