from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "similarity_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

def load_trained_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Trained model files are missing. Run: python model_training.py")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

def compare_questions(question1, question2):
    q1, q2 = clean_text(question1), clean_text(question2)
    if not q1 or not q2:
        raise ValueError("Please enter two non-empty questions.")
    model, vectorizer = load_trained_model()
    v1, v2 = vectorizer.transform([q1]), vectorizer.transform([q2])
    difference = v1 - v2
    difference.data = np.abs(difference.data)
    product = v1.multiply(v2)
    features = hstack([difference, product], format="csr")
    probabilities = model.predict_proba(features)[0]
    duplicate_index = list(model.classes_).index(1)
    probability = float(probabilities[duplicate_index] * 100)
    cosine_score = float(cosine_similarity(v1, v2)[0][0] * 100)
    if probability >= 75:
        label = "Duplicate questions"
    elif probability >= 45:
        label = "Possibly similar questions"
    else:
        label = "Different questions"
    return round(probability, 2), round(cosine_score, 2), label
