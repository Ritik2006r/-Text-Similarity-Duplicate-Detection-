from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from dataset import load_dataset
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

def make_pair_features(q1_list, q2_list, vectorizer):
    v1 = vectorizer.transform(q1_list)
    v2 = vectorizer.transform(q2_list)
    difference = v1 - v2
    difference.data = np.abs(difference.data)
    product = v1.multiply(v2)
    return hstack([difference, product], format="csr")

def train_model():
    df = load_dataset()
    df["question1"] = df["question1"].apply(clean_text)
    df["question2"] = df["question2"].apply(clean_text)
    df = df[(df["question1"] != "") & (df["question2"] != "")].copy()
    if df["is_duplicate"].nunique() != 2:
        raise ValueError("Training needs examples from both classes: 0 and 1.")
    if df["is_duplicate"].value_counts().min() < 2:
        raise ValueError("Each class needs at least 2 rows for a stratified split. Use a larger dataset.")
    X_train, X_test, y_train, y_test = train_test_split(
        df[["question1", "question2"]], df["is_duplicate"],
        test_size=0.2, random_state=42, stratify=df["is_duplicate"]
    )
    train_texts = X_train["question1"].tolist() + X_train["question2"].tolist()
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vectorizer.fit(train_texts)
    train_features = make_pair_features(X_train["question1"].tolist(), X_train["question2"].tolist(), vectorizer)
    test_features = make_pair_features(X_test["question1"].tolist(), X_test["question2"].tolist(), vectorizer)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(train_features, y_train)
    predictions = model.predict(test_features)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    print("\nMODEL EVALUATION")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "similarity_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not duplicate", "Duplicate"],
                yticklabels=["Not duplicate", "Duplicate"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    metrics = (f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\n"
               f"Recall: {recall:.4f}\nF1-score: {f1:.4f}\n"
               f"Training rows: {len(X_train)}\nTesting rows: {len(X_test)}\n")
    (REPORT_DIR / "evaluation_metrics.txt").write_text(metrics, encoding="utf-8")
    print("\nSaved model files in models/")
    print("Saved confusion matrix and metrics in reports/")
    return model, vectorizer

if __name__ == "__main__":
    train_model()
