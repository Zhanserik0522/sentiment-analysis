import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score

# Ensure project root is on sys.path so `src` package can be imported when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_dataset, preprocess_df, split_data


def stratified_sample(df, label_col, n_samples, random_state=42):
    if len(df) <= n_samples:
        return df.copy()
    # use StratifiedShuffleSplit to get a stratified subset of size n_samples
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    y = df[label_col].values
    for train_idx, _ in sss.split(df, y):
        return df.iloc[train_idx].copy()


def prepare_data(data_path, sample_size=30000, random_state=42):
    print(f"Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    print(f"Raw dataset shape: {df.shape}")
    # Normalize column names: common datasets use different names for text/label
    orig_cols = list(df.columns)
    lower_map = {c.lower(): c for c in orig_cols}
    text_col = None
    label_col = None
    for candidate in ['text', 'review', 'sentence', 'content', 'review_text', 'reviewtext']:
        if candidate in lower_map:
            text_col = lower_map[candidate]
            break
    for candidate in ['label', 'sentiment', 'sentiment_label', 'score', 'target']:
        if candidate in lower_map:
            label_col = lower_map[candidate]
            break

    if text_col and label_col:
        # rename to standardized names expected by preprocess_df
        if text_col != 'text' or label_col != 'label':
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
    else:
        raise KeyError(f"Could not find text/label columns in dataset. Available columns: {orig_cols}")

    df = preprocess_df(df)
    print(f"After preprocessing shape: {df.shape}")

    label_col = 'label'
    if sample_size and len(df) > sample_size:
        df = stratified_sample(df, label_col, sample_size, random_state=random_state)
        print(f"Stratified sampled to {len(df)} rows")

    # Compute safe fractional sizes for test and val. For large datasets we
    # aim for ~5000 samples each; for small datasets use reasonable fractions.
    total_n = len(df)
    frac_test = min(5000 / total_n, 0.2)
    frac_val = min(5000 / total_n, 0.1)
    # Ensure sum of fractions is < 0.9 to leave enough for train
    if frac_test + frac_val >= 0.9:
        # scale down val to keep room for train
        frac_val = max(0.05, 0.9 - frac_test)

    train, val, test = split_data(
        df,
        label_col=label_col,
        test_size=frac_test,
        val_size=frac_val,
        random_state=random_state,
    )
    print(f"Train/Val/Test sizes: {len(train)}/{len(val)}/{len(test)}")
    return train, val, test


def run_random_search(
    train,
    val,
    test,
    out_dir,
    n_iter=30,
    cv=5,
    random_state=42,
    quick=False,
):
    os.makedirs(out_dir, exist_ok=True)

    X_train = train['text'].values
    y_train = train['label'].values
    X_val = val['text'].values
    y_val = val['label'].values
    X_test = test['text'].values
    y_test = test['label'].values

    # We'll do two separate RandomizedSearchCV runs: LogisticRegression and RandomForest.
    # Create separate vectorizer instances for each pipeline to avoid shared state.
    tfidf_lr = TfidfVectorizer()

    # Logistic Regression pipeline (use saga solver to allow l1/l2 penalties)
    pipe_lr = Pipeline([
        ('vectorizer', tfidf_lr),
        (
            'clf',
            LogisticRegression(max_iter=1000, random_state=random_state, solver='saga')
        ),
    ])

    C_list = list(np.logspace(-4, 2, 50))
    if quick:
        tfidf_max_opts = [2000, 5000]
    else:
        tfidf_max_opts = [5000, 10000, 20000]

    param_dist_lr = {
        'vectorizer__max_features': tfidf_max_opts,
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'clf__C': C_list,
        'clf__penalty': ['l2', 'l1'],
        'clf__class_weight': [None, 'balanced'],
    }

    print("Starting RandomizedSearch for LogisticRegression...")
    rs_lr = RandomizedSearchCV(
        estimator=pipe_lr,
        param_distributions=param_dist_lr,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
    )
    t0 = time.time()
    rs_lr.fit(X_train, y_train)
    t_lr = time.time() - t0
    print(
        "LR RandomizedSearch done in "
        f"{t_lr:.1f}s. Best score: {rs_lr.best_score_}"
    )
    print(f"Best params LR: {rs_lr.best_params_}")

    best_lr = rs_lr.best_estimator_
    print(
        "LR RandomizedSearch done in "
        f"{t_lr:.1f}s. Best score: {rs_lr.best_score_}"
    )
    joblib.dump(
        best_lr,
        os.path.join(out_dir, 'best_model_lr_randomsearch.joblib')
    )

    # Evaluate on validation and test
    print("Evaluation LR on validation:")
    yv_pred = best_lr.predict(X_val)
    print(classification_report(y_val, yv_pred))
    print("Evaluation LR on test:")
    yt_pred = best_lr.predict(X_test)
    print(classification_report(y_test, yt_pred))
    acc_lr = accuracy_score(y_test, yt_pred)

    # Random Forest pipeline
    tfidf_rf = TfidfVectorizer()
    pipe_rf = Pipeline([
        ('vectorizer', tfidf_rf),
        ('clf', RandomForestClassifier(random_state=random_state, n_jobs=-1)),
    ])
    if quick:
        rf_n_estimators = [50, 100]
        tfidf_rf_opts = [2000, 5000]
    else:
        rf_n_estimators = [100, 200, 400]
        tfidf_rf_opts = [5000, 10000, 20000]

    param_dist_rf = {
        'vectorizer__max_features': tfidf_rf_opts,
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'clf__n_estimators': rf_n_estimators,
        'clf__max_depth': [None, 10, 20, 50],
        'clf__max_features': ['sqrt', 'log2', 0.2],
    }

    print("Starting RandomizedSearch for RandomForest...")
    rs_rf = RandomizedSearchCV(
        estimator=pipe_rf,
        param_distributions=param_dist_rf,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
    )
    t0 = time.time()
    rs_rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    print(
        "RF RandomizedSearch done in "
        f"{t_rf:.1f}s. Best score: {rs_rf.best_score_}"
    )
    print(f"Best params RF: {rs_rf.best_params_}")

    best_rf = rs_rf.best_estimator_
    print(
        "RF RandomizedSearch done in "
        f"{t_rf:.1f}s. Best score: {rs_rf.best_score_}"
    )
    joblib.dump(
        best_rf,
        os.path.join(out_dir, 'best_model_rf_randomsearch.joblib')
    )

    print("Evaluation RF on validation:")
    yv_pred = best_rf.predict(X_val)
    print(classification_report(y_val, yv_pred))
    print("Evaluation RF on test:")
    yt_pred = best_rf.predict(X_test)
    print(classification_report(y_test, yt_pred))
    acc_rf = accuracy_score(y_test, yt_pred)

    # Choose best by test f1 or accuracy (here accuracy as tie-breaker)
    results = {
        'lr': {
            'estimator': best_lr,
            'test_accuracy': acc_lr,
            'cv_score': rs_lr.best_score_
        },
        'rf': {
            'estimator': best_rf,
            'test_accuracy': acc_rf,
            'cv_score': rs_rf.best_score_
        }
    }

    best_key = (
        'lr'
        if results['lr']['cv_score'] >= results['rf']['cv_score']
        else 'rf'
    )
    best_overall = results[best_key]['estimator']
    joblib.dump(best_overall, os.path.join(out_dir, 'best_model_final.joblib'))

    print(f"Best overall model: {best_key}; saved to best_model_final.joblib")

    # Save a small summary CSV
    summary = pd.DataFrame([
        {
            'model': 'logistic_regression',
            'cv_score': rs_lr.best_score_,
            'test_accuracy': acc_lr
        },
        {
            'model': 'random_forest',
            'cv_score': rs_rf.best_score_,
            'test_accuracy': acc_rf
        }
    ])

    summary.to_csv(
        os.path.join(out_dir, 'random_search_summary.csv'),
        index=False
    )
    print('Summary saved to random_search_summary.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default='data/IMDB_Dataset.csv',
        help='Path to CSV dataset'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=30000,
        help='Total stratified sample size (default 30000)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=30,
        help='RandomizedSearch n_iter'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: smaller search (fewer iters, smaller RF)'
    )
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        print(
            f"Dataset not found at {data_path}. Trying fallback to "
            "data/sample_sentiment.csv"
        )
        data_path = 'data/sample_sentiment.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                'No dataset found. Place IMDB_Dataset.csv in data/ '
                'or sample_sentiment.csv'
            )

    train, val, test = prepare_data(data_path, sample_size=args.sample_size)

    # Adjust search params for quick mode
    n_iter = args.n_iter
    if args.quick:
        print('Quick mode enabled: reducing n_iter and RF complexity')
        n_iter = max(5, int(args.n_iter / 3))

    run_random_search(
        train,
        val,
        test,
        args.out_dir,
        n_iter=n_iter,
        cv=args.cv,
        quick=args.quick,
    )


if __name__ == '__main__':
    main()
