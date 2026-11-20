# 🔍 Text Similarity and Duplicate Detection

A Python and Machine Learning project that compares two text questions and predicts whether they express the same meaning. The project combines **TF-IDF vectorization**, **Logistic Regression**, and **cosine similarity**, with a simple **Streamlit** web interface.

## 📌 Project Overview

Duplicate questions are common on question-answering platforms, forums, and support websites. Manually identifying them can take time and lead to repeated content.

This project accepts two questions and provides:
- A model-estimated probability that the questions are duplicates.
- A TF-IDF cosine similarity score.
- A simple result label: Duplicate questions, Possibly similar questions, or Different questions.

> **Note:** The output is an estimate. It is not proof of plagiarism or a guarantee that two sentences have identical meanings.

## ✨ Features

- Compare two questions through a Streamlit interface.
- Clean and normalize text before processing.
- Convert text into numerical TF-IDF features.
- Train a Logistic Regression classifier on labelled question pairs.
- Calculate cosine similarity between the two question vectors.
- Preview the dataset and view its duplicate-label distribution.
- Generate evaluation metrics and a confusion-matrix image during training.

## 🧰 Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- SciPy
- Matplotlib and Seaborn
- Joblib

## 📁 Project Structure

```text
Text_Similarity_Detection/
│
├── app.py
├── dataset.py
├── preprocessing.py
├── model_training.py
├── similarity.py
├── eda.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── questions.csv
│
├── models/
│   └── (generated model files)
│
└── reports/
    └── (generated charts and evaluation report)
```

## 📊 Dataset

The project reads the CSV file from:

`dataset/questions.csv`

The CSV must contain these three columns:

| Column | Description |
|---|---|
| `question1` | First question in the pair |
| `question2` | Second question in the pair |
| `is_duplicate` | Target label: `1` for duplicate and `0` for not duplicate |

Extra columns in the original Quora Question Pairs format, such as `id`, `qid1`, and `qid2`, are not required.

**Dataset limitation:** The repository currently includes only 12 sample rows so that the project structure and pipeline can be tested. This small sample is not sufficient for reliable model evaluation or meaningful accuracy claims. For a proper experiment, replace it with a larger, correctly labelled question-pairs dataset that has examples from both classes (0 and 1).

## ⚙️ How It Works

1. **Dataset loading:** `dataset.py` reads the CSV, checks required columns, removes unusable rows, and validates labels.
2. **Text preprocessing:** `preprocessing.py` converts text to lowercase, removes URLs, HTML tags and punctuation, and normalizes whitespace.
3. **Exploratory Data Analysis:** `eda.py` prints basic dataset information and creates plots for label distribution and question word counts.
4. **Feature extraction:** `model_training.py` fits a TF-IDF vectorizer on training questions. It uses unigrams and bigrams.
5. **Pair feature creation:** The model receives the absolute difference and element-wise product of the two TF-IDF vectors.
6. **Model training:** Logistic Regression is trained using a stratified train/test split.
7. **Prediction:** `similarity.py` loads the saved model and vectorizer, predicts duplicate probability, and calculates cosine similarity.
8. **User interface:** `app.py` displays the comparison results and dataset preview in Streamlit.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ritik2006r/-Text-Similarity-Duplicate-Detection-.git
```

### 2. Open the project folder

```bash
cd -Text-Similarity-Duplicate-Detection-/Text_Similarity_Detection
```

### 3. Create and activate a virtual environment (Windows PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell does not allow environment activation, you can continue using `python -m pip` without activating it.

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## ▶️ Run the Project

Run these commands from the `Text_Similarity_Detection` folder.

### 1. Check that the dataset loads

```bash
python dataset.py
```

### 2. Generate EDA charts

```bash
python eda.py
```

### 3. Train the machine learning model

```bash
python model_training.py
```

This creates the following files:

- `models/similarity_model.pkl`
- `models/tfidf_vectorizer.pkl`
- `reports/confusion_matrix.png`
- `reports/evaluation_metrics.txt`

### 4. Launch the Streamlit application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

## 📈 Model Evaluation

The training script reports:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

No performance score is claimed in this README because the included dataset is only a small sample. Train and evaluate the model using a larger labelled dataset before reporting results.

## 📂 Main Python Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `dataset.py` | Dataset loading and validation |
| `preprocessing.py` | Text cleaning and normalization |
| `eda.py` | Exploratory data analysis and plots |
| `model_training.py` | TF-IDF feature extraction, training, and evaluation |
| `similarity.py` | Loads the trained model and compares two questions |
| `requirements.txt` | Python package dependencies |

## ⚠️ Limitations

- The included dataset is very small and is only intended as a pipeline sample.
- Prediction quality depends heavily on the size, quality, and balance of the labelled training data.
- TF-IDF is based on word and phrase occurrence; it may not understand every semantic relationship.
- The model is an educational baseline and should not be treated as a production plagiarism or moderation system.

## 🚀 Future Improvements

- Train and evaluate using a larger question-pairs dataset.
- Experiment with advanced sentence embeddings and transformer-based models.
- Add more detailed visualizations and error analysis.
- Deploy the Streamlit application for public access.

## 👨‍💻 Author

**Ritik**

---

If you find this project useful, feel free to star the repository.
