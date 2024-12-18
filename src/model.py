import argparse
import gzip
import json
import joblib
from typing import Iterable
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load labels
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')

class Model:
    def __init__(self):
        """
        Initializes the Model class by setting up the machine learning pipeline and parameter grid.
        """
        self.pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(solver='liblinear')),
        ])
        self.param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l1', 'l2']
        }
        self.model = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='f1_weighted')


    def train(self, train_data: Iterable[dict]):
        """
        Trains the model using the provided training data.

        Args:
            train_data (Iterable[dict]): The training data, where each dictionary contains 'text' and 'label' keys.
        """
        texts = [x["text"] for x in train_data]
        labels = [x["label"] for x in train_data]
        self.model.fit(texts, labels)
        print(f"Best parameters: {self.model.best_params_}")

    def predict(self, data: Iterable[dict]):
        """
        Makes predictions using the trained model.

        Args:
            data (Iterable[dict]): The input data, where each dictionary contains a 'text' key.

        Returns:
            list: The predicted labels for the input data.
        """
        texts = [x["text"] for x in data]
        return self.model.predict(texts)
    
    def evaluate(self, test_data: Iterable[dict]):
        """
        Evaluates the model using the provided test data.

        Args:
            test_data (Iterable[dict]): The test data, where each dictionary contains 'text' and 'label' keys.

        Returns:
            None
        """
        texts = [x["text"] for x in test_data]
        true_labels = [x["label"] for x in test_data]
        predictions = self.model.predict(texts)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    def save(self, path: str):
        """
        Saves the trained model to the specified path.

        Args:
            path (str): The path to save the model.
        """
        with open(path, 'wb') as f:
            joblib.dump(self.model, f)

    def load(self, path: str):
        """
        Loads the trained model from the specified path.

        Args:
            path (str): The path to load the model from.
        """
        with open(path, 'rb') as f:
            self.model = joblib.load(f)

def load_dataset(path):
    """
    Loads a dataset from a gzip-compressed JSON lines file.

    Args:
        path (str): The path to the gzip-compressed JSON lines file.

    Returns:
        list: A list of dictionaries representing the dataset.
    """
    data = []
    with gzip.open(path, "rb") as f_in:
        for line in f_in:
            data.append(json.loads(line))
    return data


def main(args):
    train_data = load_dataset(args.train)
    model = Model()
    model.train(train_data)

    # Save the trained model
    model.save(model_path)

    test_data = load_dataset(args.test)

    example = test_data[0]
    prediction = model.predict([example])[0]
    print("\nExample:")
    print(example)
    print(f"\nPredicted label: {prediction}")

    # Evaluate the model
    model.evaluate(test_data)

    # Load the model and re-evaluate to ensure it works
    model.load(model_path)
    model.evaluate(test_data)


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default="data/train.jsonl.gz",
        help="path to training data (.jsonl.gz file)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/test.jsonl.gz",
        help="path to test data (.jsonl.gz file)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
