from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_dataset

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"

def perform_eda():
    df = load_dataset()
    REPORT_DIR.mkdir(exist_ok=True)
    print("Dataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate label counts:")
    print(df["is_duplicate"].value_counts().sort_index())
    df["question1_word_count"] = df["question1"].str.split().str.len()
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="is_duplicate")
    plt.title("Duplicate vs Non-Duplicate Question Pairs")
    plt.xlabel("Label (0 = Not duplicate, 1 = Duplicate)")
    plt.ylabel("Number of pairs")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "duplicate_distribution.png", dpi=150)
    plt.close()
    plt.figure(figsize=(7, 5))
    sns.histplot(df["question1_word_count"], bins=30, kde=True)
    plt.title("Question 1 Word Count Distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Number of questions")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "question_word_count.png", dpi=150)
    plt.close()
    print("\nEDA graphs saved in reports/")

if __name__ == "__main__":
    perform_eda()
