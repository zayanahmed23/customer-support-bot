import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
from pathlib import Path

# Paths
DATA_PATH = Path("data/intents.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    print("DATA_PATH:", DATA_PATH)
    print("Exists?:", DATA_PATH.exists())
    df = pd.read_csv(DATA_PATH)
    # columns: text, intent
    return df["text"], df["intent"]

def train():
    X, y = load_data()

    # Convert ALL text to TF-IDF features (no train_test_split here)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),   # unigrams and bigrams
        stop_words="english"
    )
    X_vec = vectorizer.fit_transform(X)

    # Train classifier on ALL data
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y)

    # Optional: see how well it fits the training data
    y_pred = clf.predict(X_vec)
    print("Training set performance (on all data):")
    print(classification_report(y, y_pred))

    # Save model and vectorizer
    with open(MODEL_DIR / "intent_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open(MODEL_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nSaved model to 'models/intent_classifier.pkl'")
    print("Saved vectorizer to 'models/vectorizer.pkl'")

if __name__ == "__main__":
    train()
