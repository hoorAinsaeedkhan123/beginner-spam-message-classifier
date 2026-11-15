"""Train and use a beginner-friendly spam message classifier."""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "messages.csv"


def load_data(path=DATA_PATH):
    """Load messages and perform basic data checks."""
    data = pd.read_csv(path)
    required_columns = {"message", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("CSV must contain 'message' and 'label' columns.")

    data = data.dropna(subset=["message", "label"]).copy()
    data["message"] = data["message"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["message"] != ""]
    data = data[data["label"].isin(["spam", "ham"])]

    if data.empty:
        raise ValueError("No valid messages found in the dataset.")
    if data["label"].nunique() < 2:
        raise ValueError("The dataset must contain both spam and ham labels.")
    return data


def build_model():
    """Create a simple text-counting + Naive Bayes machine-learning pipeline."""
    return Pipeline([
        ("word_counts", CountVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB())
    ])


def train_and_evaluate(data):
    """Split data, train the model, and return evaluation results."""
    x_train, x_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"]
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["ham", "spam"]
        ),
        "classification_report": classification_report(
            y_test, predictions, labels=["ham", "spam"], zero_division=0
        ),
        "test_size": len(y_test)
    }
    return model, results


def cross_validate(data, folds=5, repeats=5):
    """Return mean and standard deviation of accuracy over repeated k-fold splits.

    With only ~100 messages, one 25-message test split is very noisy, so this
    gives a more trustworthy estimate than a single split.
    """
    splitter = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=42
    )
    scores = cross_val_score(
        build_model(), data["message"], data["label"], cv=splitter
    )
    return scores.mean(), scores.std()


def predict_message(model, message):
    """Predict whether one message is spam or ham."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a non-empty message.")
    prediction = model.predict([message.strip()])[0]
    return "Spam" if prediction == "spam" else "Not spam"


def main():
    try:
        data = load_data()
        model, results = train_and_evaluate(data)

        print("\n=== Beginner Spam Message Classifier ===")
        print(f"Messages loaded: {len(data)}")
        print("Training/testing split: 75% / 25%")
        print(f"Test messages: {results['test_size']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print("\nConfusion matrix (rows=actual, columns=predicted):")
        print("                 Predicted ham   Predicted spam")
        matrix = results["confusion_matrix"]
        print(f"Actual ham       {matrix[0, 0]:14d}   {matrix[0, 1]:14d}")
        print(f"Actual spam      {matrix[1, 0]:14d}   {matrix[1, 1]:14d}")
        print("\nClassification report:")
        print(results["classification_report"])

        cv_mean, cv_std = cross_validate(data)
        print(f"Cross-validated accuracy (5-fold x 5): {cv_mean:.2%} +/- {cv_std:.2%}")
        print("Note: the dataset is tiny, so a single split can be misleading.")

        print("\nTry your own messages. Type 'quit' to stop.")
        while True:
            try:
                message = input("\nEnter a message: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if message.lower() == "quit":
                print("Goodbye!")
                break
            if not message:
                print("Please type a message first.")
                continue
            print("Prediction:", predict_message(model, message))

    except FileNotFoundError:
        print("Dataset not found. Make sure data/messages.csv exists.")
    except (ValueError, pd.errors.ParserError) as error:
        print("Could not process the dataset:", error)


if __name__ == "__main__":
    main()
