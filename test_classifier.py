import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier import load_data, build_model, train_and_evaluate, predict_message, cross_validate


class TestSpamClassifier(unittest.TestCase):
    def test_dataset_has_both_classes(self):
        data = load_data()
        self.assertGreater(len(data), 0)
        self.assertEqual(set(data["label"].unique()), {"ham", "spam"})

    def test_model_trains_and_returns_accuracy(self):
        data = load_data()
        model, results = train_and_evaluate(data)
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertGreater(results["test_size"], 0)

    def test_prediction_is_readable(self):
        data = load_data()
        model, _ = train_and_evaluate(data)
        result = predict_message(model, "Congratulations! Claim your free prize now")
        self.assertIn(result, {"Spam", "Not spam"})

    def test_empty_message_is_rejected(self):
        model = build_model()
        data = load_data()
        model.fit(data["message"], data["label"])
        with self.assertRaises(ValueError):
            predict_message(model, "   ")

    def test_obvious_messages_are_classified(self):
        data = load_data()
        model, _ = train_and_evaluate(data)
        self.assertEqual(
            predict_message(model, "Claim your free cash prize now! Click this link"),
            "Spam",
        )
        self.assertEqual(
            predict_message(model, "Can you send me the lecture notes before class?"),
            "Not spam",
        )

    def test_cross_validation_returns_valid_scores(self):
        mean, std = cross_validate(load_data(), folds=3, repeats=1)
        self.assertTrue(0.0 <= mean <= 1.0)
        self.assertGreaterEqual(std, 0.0)

    def test_load_data_rejects_missing_columns(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.csv")
            with open(bad, "w") as f:
                f.write("text,tag\nhello,ham\n")
            with self.assertRaises(ValueError):
                load_data(bad)


if __name__ == "__main__":
    unittest.main()
