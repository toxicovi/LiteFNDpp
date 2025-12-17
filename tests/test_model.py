import unittest
from litefndpp.model import LiteFNDpp

class TestLiteFNDpp(unittest.TestCase):
    def setUp(self):
        # Sample texts and labels
        self.texts = [
            "Breaking news! Scientists discovered a shocking cure for COVID.",
            "Local news reports a minor traffic accident downtown."
        ]
        self.labels = [1, 0]  # 1 = Fake, 0 = Real
        self.model = LiteFNDpp()
    
    def test_training_and_prediction(self):
        # Train the model
        self.model.fit(self.texts, self.labels)
        preds = self.model.predict(self.texts)
        self.assertEqual(len(preds), len(self.texts))
        # Predictions should be 0 or 1
        self.assertTrue(all(p in [0,1] for p in preds))
    
    def test_explanation_output(self):
        explanation = self.model.explain(self.texts[0])
        self.assertIn("top_features", explanation)
        self.assertIn("cognitive_scores", explanation)
        self.assertIn("prediction", explanation)
        self.assertIsInstance(explanation["cognitive_scores"], dict)

if __name__ == "__main__":
    unittest.main()
