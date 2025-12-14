import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import warnings

# ensure stopwords are available
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs/mentions/punctuation and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return ' '.join(tokens)


def clean_text_negation(text: str) -> str:
    """Negation-aware cleaning.

    Converts tokens after a negation word into a negated token (prefix with "not_")
    until the next punctuation/stop boundary. This is a lightweight heuristic that
    helps capture constructs like "not good" -> "not_good" so models can learn
    negated contexts.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    # replace punctuation with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    negation_tokens = set(['not', "don't", "didn't", 'never', "no", "can't", "cannot", "won't"])
    out = []
    negate = False
    for t in text.split():
        if t in negation_tokens:
            # keep the negation word itself
            out.append(t)
            negate = True
            continue
        if negate:
            # attach negation prefix and stop negation after one token to avoid long spans
            nt = f'not_{t}'
            if nt not in STOPWORDS:
                out.append(nt)
            negate = False
        else:
            if t not in STOPWORDS:
                out.append(t)
    return ' '.join(out)


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset into a DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess_df(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label') -> pd.DataFrame:
    """Keep only text and label columns, fill NA, and clean text."""
    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna('').apply(clean_text)
    df = df[df[text_col].str.strip() != '']
    return df


def split_data(df: pd.DataFrame, label_col: str = 'label', test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """Split dataframe into train/val/test.

    Attempts stratified splits first. If dataset is too small for stratification
    (which raises ValueError), falls back to non-stratified splits and issues a warning.
    Returns train, val, test DataFrames with reset indices.
    """
    # First, try stratified test split
    try:
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df[label_col], random_state=random_state
        )
    except ValueError:
        warnings.warn(
            'Stratified split for test failed (too few samples per class). Using non-stratified split.'
        )
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

    # Then split train and val. Compute relative val size with respect to train_val
    val_relative = val_size / (1 - test_size)
    try:
        train, val = train_test_split(
            train_val, test_size=val_relative, stratify=train_val[label_col], random_state=random_state
        )
    except ValueError:
        warnings.warn(
            'Stratified split for validation failed (too few samples per class). Using non-stratified split.'
        )
        train, val = train_test_split(
            train_val, test_size=val_relative, random_state=random_state
        )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
