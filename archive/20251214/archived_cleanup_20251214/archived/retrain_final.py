import os
import sys
import joblib
import argparse

# Ensure project root on path before importing local package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_dataset, preprocess_df, split_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score



def retrain(data_path, model_path, out_path, random_state=42):
    print(f"Loading dataset from {data_path}")
    df = load_dataset(data_path)

    # Normalize columns like in run_random_search
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
        if text_col != 'text' or label_col != 'label':
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
    else:
        raise KeyError(f"Could not find text/label columns in dataset. Available columns: {orig_cols}")

    df = preprocess_df(df)
    print(f"Preprocessed shape: {df.shape}")

    # Recreate split with fixed random state: use split_data to get same test set sizes
    train, val, test = split_data(df, label_col='label', test_size=0.1, val_size=0.1, random_state=random_state)
    print(f"Train/Val/Test sizes: {len(train)}/{len(val)}/{len(test)}")

    # Load model
    print(f"Loading estimator from {model_path}")
    clf = joblib.load(model_path)

    # Combine train+val
    train_val = pd.concat([train, val]).reset_index(drop=True)
    X_train_val = train_val['text'].values
    y_train_val = train_val['label'].values

    print("Retraining estimator on train+val...")
    # If estimator is a Pipeline, call fit on it
    clf.fit(X_train_val, y_train_val)

    # Evaluate on test
    X_test = test['text'].values
    y_test = test['label'].values
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save retrained model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"Saved retrained model to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/IMDB_Dataset.csv')
    parser.add_argument('--model', type=str, default='models/best_model_final.joblib')
    parser.add_argument('--out', type=str, default='models/best_model_final_trained.joblib')
    args = parser.parse_args()

    retrain(args.data, args.model, args.out)


if __name__ == '__main__':
    import pandas as pd
    main()
