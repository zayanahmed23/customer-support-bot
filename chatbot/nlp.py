import pickle
from pathlib import Path

MODEL_DIR = Path("models")

# Load the trained model and vectorizer once, when this module is imported
with open(MODEL_DIR / "intent_classifier.pkl", "rb") as f:
    INTENT_MODEL = pickle.load(f)

with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
    VECTORIZER = pickle.load(f)


def predict_intent(user_text: str, threshold: float = 0.3) -> str:
    """
    Predict the intent of the user's message.
    If the model's confidence is too low, return 'fallback'.
    """
    X_vec = VECTORIZER.transform([user_text])

    # If the model supports probabilities, use them to decide fallback
    if hasattr(INTENT_MODEL, "predict_proba"):
        probs = INTENT_MODEL.predict_proba(X_vec)[0]
        max_prob = probs.max()
        intent = INTENT_MODEL.classes_[probs.argmax()]
        # Low confidence -> fallback
        if max_prob < threshold:
            return "fallback"
        return intent
    else:
        # No probability info available
        return INTENT_MODEL.predict(X_vec)[0]
