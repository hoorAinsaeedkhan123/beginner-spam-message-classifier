# Beginner Spam Message Classifier

A small, beginner-friendly machine-learning project that predicts whether a text message is **spam** or **not spam**.

It uses two easy ideas:
1. **Bag of Words:** count how often words appear in messages.
2. **Naive Bayes:** use word patterns learned from labelled examples to make a prediction.

> Learning project: the included dataset is small and illustrative. Its accuracy is not a measure of real-world spam detection performance. Do not use this model to make important decisions or treat its predictions as certain.

## Features

- Loads messages from a CSV file.
- Checks for missing or invalid data.
- Splits examples into training and testing sets.
- Converts text into word counts.
- Trains a Multinomial Naive Bayes model.
- Reports accuracy, a confusion matrix, and precision/recall/F1.
- Adds repeated cross-validation, because one 25-message test split is very noisy.
- Lets you type messages and see predictions.
- Includes basic automated tests.

## Project structure

```text
beginner-spam-message-classifier/
├── data/
│   └── messages.csv
├── src/
│   └── classifier.py
├── tests/
│   └── test_classifier.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- pip

## Setup

Open a terminal in the project folder.

**Windows**
```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the project

```bash
python src/classifier.py
```

Type a message when prompted. Type `quit` to exit.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The interactive prompt also exits cleanly on Ctrl+C / Ctrl+D.

## How it works

1. The CSV contains example messages and their labels: `spam` or `ham` (`ham` means not spam).
2. `train_test_split` keeps some messages aside for testing.
3. `CountVectorizer` turns text into numbers by counting words.
4. `MultinomialNB` learns word patterns from the training messages.
5. The test set checks how well the model predicts messages it did not train on.
6. The program can then predict a new message typed by the user.

## Understanding the evaluation

- **Accuracy:** the fraction of test messages classified correctly.
- **Precision:** when the model predicts spam, how often that prediction is correct.
- **Recall:** how many actual spam messages the model finds.
- **F1-score:** a balance between precision and recall.
- **Confusion matrix:** shows correct and incorrect predictions for each class.

On the included data, a single 75/25 split gives about 80% accuracy while repeated 5-fold cross-validation gives about 92% (+/- 5%). That gap shows how unreliable a single split is on 100 messages. The dataset is intentionally small so a beginner can read it. A larger, diverse, carefully labelled dataset is needed for meaningful performance evaluation. A high score on this tiny dataset does not mean the model will reliably detect real spam.

## Easy improvements to try

- Add more labelled examples from a properly licensed dataset.
- Compare word counts with TF-IDF features.
- Try a different train/test split and compare results.
- Add a simple web interface after understanding the command-line version.

## License

This project is provided for learning and portfolio practice. If you publish it, you can add an MIT license after confirming you want to release your own contributions under those terms.
