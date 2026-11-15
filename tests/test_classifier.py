import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier import load_data, build_model, train_and_evaluate, predict_message


class TestSpamClassifier(unittest.TestCase):

    def test_dataset(self):
        data = load_data()
        self.assertEqual(set(data["label"].unique()), {"ham", "spam"})

    def test_model(self):
        data = load_data()
        model, results = train_and_evaluate(data)
        self.assertGreaterEqual(results["accuracy"], 0)

    def test_prediction(self):
        data = load_data()
        model, _ = train_and_evaluate(data)
        result = predict_message(model, "Congratulations! Claim your free prize")
        self.assertIn(result, {"Spam", "Not spam"})

    def test_empty_message(self):
        data = load_data()
        model = build_model()
        model.fit(data["message"], data["label"])

        with self.assertRaises(ValueError):
            predict_message(model, "")


if __name__ == "__main__":
    unittest.main()
