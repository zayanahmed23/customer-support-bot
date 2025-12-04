from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/faq.csv")

# Load FAQ data once
_df = pd.read_csv(DATA_PATH)

# Drop any rows where question or answer is missing
_df = _df.dropna(subset=["question", "answer"])

# Make sure both columns are strings
_df["question"] = _df["question"].astype(str)
_df["answer"] = _df["answer"].astype(str)

FAQ_QUESTIONS = _df["question"].tolist()
FAQ_ANSWERS = _df["answer"].tolist()

# Build a TF-IDF vectorizer for FAQ questions
_FAQ_VECTORIZER = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english"
)
FAQ_MATRIX = _FAQ_VECTORIZER.fit_transform(FAQ_QUESTIONS)


def semantic_faq_search(query: str, threshold: float = 0.25) -> str | None:
    """
    Return the best matching FAQ answer for the given query
    using cosine similarity. If similarity is below threshold,
    return None.
    """
    if not query or not query.strip():
        return None

    query_vec = _FAQ_VECTORIZER.transform([query])
    sims = cosine_similarity(query_vec, FAQ_MATRIX)[0]

    best_idx = sims.argmax()
    best_score = sims[best_idx]

    if best_score < threshold:
        return None

    return FAQ_ANSWERS[best_idx]
