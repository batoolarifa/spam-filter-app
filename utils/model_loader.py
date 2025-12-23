import pickle

def load_models(vectorizer_path="saved_models/vectorizer_TF-IDF.pkl",
                model_path="saved_models/SVM_TF-IDF.pkl"):
    """Load vectorizer and SVM model."""
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model
