import sys
import pathlib
import streamlit as st
import joblib

# Ensure project root is on sys.path so `src` package imports work when
# running the app via `streamlit run` from any CWD.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_text

st.title('Sentiment Analysis Demo')

@st.cache_resource
def load():
    try:
        clf = joblib.load('models/best_model.joblib')
    except Exception:
        clf = None
    return clf

clf = load()

text = st.text_area('Enter text to analyze')

if st.button('Predict'):
    if not text.strip():
        st.warning('Please enter some text')
    elif clf is None:
        st.error('Model not found. Train models first or place model file at models/best_model.joblib')
    else:
        cleaned = clean_text(text)
        pred = clf.predict([cleaned])[0]
        st.write('Predicted sentiment:', pred)
