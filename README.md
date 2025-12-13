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

- If `streamlit` is not recognized after activation, run it with the full path:

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
