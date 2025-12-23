from .preprocessing import preprocess_text

def predict(text: str, vectorizer, model) -> str:
    """Preprocess text, vectorize, and predict Spam/Ham."""
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    result = model.predict(vectorized)[0]
    return "Spam" if result == 1 else "Not Spam"
