import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib


def build_tfidf_vectorizer(max_features: int = 10000):
    return TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2)
    )


def train_logistic_regression(X_train, y_train, X_val=None, y_val=None):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = (
        precision_recall_fscore_support(y, y_pred, average='weighted')
    )
    # roc_auc for multiclass: use 'ovr' strategy with label binarization
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def save_model(clf, path: str):
    joblib.dump(clf, path)


def load_model(path: str):
    return joblib.load(path)
