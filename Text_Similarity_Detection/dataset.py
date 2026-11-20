from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "dataset" / "questions.csv"
REQUIRED_COLUMNS = ["question1", "question2", "is_duplicate"]

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df["question1"] = df["question1"].fillna("").astype(str).str.strip()
    df["question2"] = df["question2"].fillna("").astype(str).str.strip()
    df["is_duplicate"] = pd.to_numeric(df["is_duplicate"], errors="coerce")
    df = df.dropna(subset=["is_duplicate"])
    df = df[df["is_duplicate"].isin([0, 1])]
    df = df[(df["question1"] != "") & (df["question2"] != "")]
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    if df.empty:
        raise ValueError("No usable question pairs found.")
    return df.reset_index(drop=True)

if __name__ == "__main__":
    data = load_dataset()
    print("Dataset loaded:", data.shape)
    print(data.head())
    print(data["is_duplicate"].value_counts().sort_index())
