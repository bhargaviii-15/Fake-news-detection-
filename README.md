# Fake News Detection

**Repository / Project:** Fake News Detection

**Notebook:** `/mnt/data/Fake_News_Detection.ipynb`

---

## Overview

This project implements a fake news detection pipeline using classical ML / NLP techniques (data cleaning, feature extraction, model training, evaluation). The primary work is contained in the Jupyter notebook `Fake_News_Detection.ipynb`.

Goals:
- Preprocess and explore a news dataset
- Extract textual features (TF-IDF / embeddings)
- Train and evaluate classification models (Logistic Regression, Random Forest, etc.)
- Export a simple inference function for predictions


## Repository structure (suggested)

```
/ (root)
├─ data/
│  ├─ raw/               # original datasets (CSV/JSON)
│  └─ processed/         # cleaned/processed CSVs
├─ notebooks/
│  └─ Fake_News_Detection.ipynb
├─ src/
│  ├─ data_processing.py
│  ├─ features.py
│  ├─ train.py
│  └─ predict.py
├─ models/               # trained model files (pickle / joblib)
├─ requirements.txt
└─ README.md
```


## Requirements

Create a virtual environment and install dependencies. Example `requirements.txt` (add exact versions you used):

```
python>=3.8
pandas
numpy
scikit-learn
nltk
joblib
jupyter
matplotlib
seaborn
``` 

Install:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## How to run

### 1) Open the notebook (recommended)

```bash
jupyter notebook /mnt/data/Fake_News_Detection.ipynb
# or
jupyter lab /mnt/data/Fake_News_Detection.ipynb
```

Work through the notebook cells to reproduce preprocessing, training, and evaluation results.


### 2) CLI / script usage (optional)

If you move pipeline code out of the notebook into `src/`, example train and predict commands:

```bash
python src/train.py --data data/processed/train.csv --out models/best_model.joblib
python src/predict.py --model models/best_model.joblib --text "This is a sample news article" 
```

Example minimal `predict.py` content (paste into `src/predict.py`):

```python
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = 'models/best_model.joblib'
VECT_PATH = 'models/tfidf_vectorizer.joblib'

if __name__ == '__main__':
    text = ' '.join(sys.argv[1:])
    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECT_PATH)
    X = vect.transform([text])
    pred = model.predict(X)
    print('Prediction:', pred[0])
```


## Dataset

- Place raw datasets under `data/raw/` and processed files under `data/processed/`.
- Keep a short README in `data/` describing the original data source, license, and preprocessing steps.


## Preprocessing & Feature Extraction

Typical steps implemented in the notebook:
- Lowercasing, punctuation removal
- Tokenization and stopword removal (NLTK)
- Optional lemmatization
- TF-IDF vectorization (or alternative embeddings)


## Models & Evaluation

Common models to try:
- Logistic Regression
- Random Forest
- SVM

Evaluation metrics included in the notebook:
- Accuracy
- Precision, Recall, F1-score
- Confusion matrix


## Tips & Next steps

- Try class balancing techniques (SMOTE, class weights) if your dataset is imbalanced.
- Experiment with transformer-based embeddings (BERT) for improved accuracy.
- Add a small REST API (Flask/FastAPI) around `predict.py` for deployment.




