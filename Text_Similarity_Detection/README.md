# Text Similarity and Duplicate Detection

A beginner-friendly Python project that classifies whether two questions are duplicates.

## Dataset

Place your Quora Question Pairs CSV at:

`dataset/questions.csv`

The CSV must contain these columns:

- `question1`
- `question2`
- `is_duplicate` (0 = not duplicate, 1 = duplicate)

The original Quora CSV can also contain `id`, `qid1`, and `qid2`; the project ignores those extra columns.

**Important:** The included `questions.csv` contains only 12 sample rows to check that the code runs. It is not large enough for meaningful model evaluation. Replace it with a properly labelled larger dataset before drawing conclusions from accuracy or other metrics.

## Setup (Windows / VS Code)

Open a terminal in this project folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can run `python -m pip install -r requirements.txt` without activating the environment.

## Run in this order

```bash
python dataset.py
python eda.py
python model_training.py
python -m streamlit run app.py
```

- `dataset.py`: loads and validates the CSV.
- `eda.py`: creates charts in `reports/`.
- `model_training.py`: trains TF-IDF + Logistic Regression, evaluates it, and saves model files.
- `app.py`: launches the Streamlit comparison interface.

## Generated files

`models/` is populated after model training:
- `similarity_model.pkl`
- `tfidf_vectorizer.pkl`

`reports/` is populated after EDA and training:
- `duplicate_distribution.png`
- `question_word_count.png`
- `confusion_matrix.png`
- `evaluation_metrics.txt`

Generated model binaries and reports are excluded from Git using `.gitignore`. Run the training and EDA scripts locally to generate them.

## Notes

- Use a larger labelled dataset for meaningful metrics.
- The train/test split is stratified and uses a fixed random seed for repeatability.
- This is an educational baseline, not a production plagiarism detector.
