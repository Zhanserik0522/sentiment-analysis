# Sentiment Analysis — Ready-to-run

This repository is an end-to-end sentiment analysis demo. It contains a small example dataset, a notebook with preprocessing and model training, a simple Streamlit app to try predictions locally, and helper scripts.

What you get
- `data/sample_sentiment.csv` — tiny example (text,label) to test the code.
- `notebooks/Sentiment_Analysis.ipynb` — walkthrough: EDA, preprocessing, training, evaluation.
- `src/` — utility modules (preprocessing, models).
- `models/best_model.joblib` — example pipeline (TF-IDF + classifier).
- `app/app.py` — Streamlit demo for quick manual testing.
- `scripts/test_predict.py` — small script that loads the saved pipeline and runs sample predictions.

Quick Windows (PowerShell) setup
1) Open PowerShell and go to the project folder.

2) Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Notes:
- If `Activate.ps1` is blocked by policy, run PowerShell as Administrator and allow scripts, or execute: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.
- If `pip install` fails with a long-path error on Windows, move the project to a shorter path (for example `C:\projects\Final`) or enable long paths in Windows settings.

Run the notebook
- Open `notebooks/Sentiment_Analysis.ipynb` in Jupyter or VS Code and run cells step-by-step to train/evaluate models.

Run the Streamlit demo
- Recommended (after activating venv):

```powershell
streamlit run app\app.py
```

- If `streamlit` is not recognized after 
activation, run it with the full path:

```powershell
.\venv\Scripts\streamlit.exe run app\app.py
```

Quick test (command-line)
- Run the small test script (uses the saved pipeline):

```powershell
.\venv\Scripts\python.exe scripts\test_predict.py
```

Where the model is
- The demo uses `models/best_model.joblib` (a pipeline: vectorizer + classifier). Replace it with your trained model if you retrain.

Common troubleshooting
- `ModuleNotFoundError`: make sure the venv is activated and you installed requirements.
- `streamlit` command not found: either activate venv or call the full path to `streamlit.exe` in the venv's `Scripts` folder.
- `OSError` on package install (long paths): move the project to a shorter path or enable long paths in Windows.
- `InconsistentVersionWarning` from sklearn when loading a saved model: re-train/save the model with the same scikit-learn version installed in the venv to avoid warnings.

How to retrain
- Open the notebook `notebooks/Sentiment_Analysis.ipynb`, follow EDA and training cells, and save a new pipeline to `models/best_model.joblib`.

Running the automated hyperparameter search and final retraining
- RandomizedSearch (example): runs an expanded randomized search over LogisticRegression and RandomForest on a stratified sample and saves best models to `models/`.
	- Recommended from PowerShell (after activating venv):

```powershell
# Run randomized search on a 30k stratified sample (n_iter=30, CV=5). This may take hours depending on CPU.
.\venv\Scripts\python.exe scripts\run_random_search.py --data data/IMDB_Dataset.csv --sample-size 30000 --out-dir models --n-iter 30 --cv 5
```

- Retrain final model on full train+val and evaluate on test (fast after selecting best params):

```powershell
.\venv\Scripts\python.exe scripts\retrain_final.py --data data/IMDB_Dataset.csv --model models/best_model_final.joblib --out models/best_model_final_trained.joblib
```

Notes on resources and time
- The randomized search can be CPU- and memory-intensive. Recommended options to speed up:
	- Reduce `--n-iter` (e.g., 10–20) or `tfidf__max_features` in the search.
	- Limit `n_estimators` for RandomForest during search (100–200).
	- Use `n_jobs=-1` to parallelize across cores (already used in scripts). If your machine has limited cores, reduce `n_jobs`.

- For final training (retrain_final), time is proportional to the chosen vectorizer size and estimator; LogisticRegression retrain is usually fast (<5 min), RandomForest may take longer depending on `n_estimators`.

Dataset (IMDB) and recommended workflow
Dataset source: IMDB Dataset of 50K Movie Reviews on Kaggle: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Short description:

Key variables:

Recommended download method (Kaggle CLI):
1) Install Kaggle CLI and put your `kaggle.json` (API token) into `%USERPROFILE%\.kaggle\kaggle.json`.

```powershell
pip install kaggle
# place kaggle.json in %USERPROFILE%\.kaggle\kaggle.json
```
# Sentiment Analysis — Quick start

This repository is a compact sentiment analysis demo. It includes a Streamlit demo, helper scripts, and example training utilities.

Quick start (Windows PowerShell)
1) Create and activate a virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Run the Streamlit demo:

```powershell
streamlit run app\app.py
```

If `streamlit` is not found after activation, run:

```powershell
.\venv\Scripts\streamlit.exe run app\app.py
```

Quick test (command-line)

```powershell
.\venv\Scripts\python.exe scripts\test_predict.py
```

Where the model is
- The app loads `models/model_final.joblib` by default (replace it with your own trained model if needed).

Reproduce final training (optional)
- If you want to retrain the augmented final model, use:

```powershell
python scripts/train_final_imdb_augmented.py --data data/IMDB_Dataset.csv --out models/model_final_imdb_augmented.joblib
```

Notes
- Large datasets (IMDB) are not included; put `data/IMDB_Dataset.csv` into `data/` if you plan to retrain.
- If PowerShell blocks scripts, run PowerShell as Administrator and use `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

If you want the README shorter or in Russian, tell me which parts to keep and I will update it.


- There is no automatic checkpointing for RandomizedSearchCV in this repo. To safely restart a search:
