import sys
from pathlib import Path
import joblib
import argparse

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.preprocessing import load_dataset, preprocess_df, split_data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def main(data_path='data/IMDB_Dataset.csv', out_path='models/model_final_imdb.joblib', max_features=20000, C=1.0):
    DATA = PROJECT / data_path
    if not DATA.exists():
        raise SystemExit(f'IMDB dataset not found at {DATA}. Place IMDB_Dataset.csv in data/')
    print('Loading IMDB dataset...')
    df = load_dataset(str(DATA))
    # normalize columns
    if 'review' in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
    df = preprocess_df(df)
    train, val, test = split_data(df, test_size=0.1, val_size=0.1, random_state=42)
    print(f'Train/Val/Test sizes: {len(train)}/{len(val)}/{len(test)}')

    X_train = train['text'].tolist()
    y_train = train['label'].tolist()
    X_val = val['text'].tolist()
    y_val = val['label'].tolist()
    X_test = test['text'].tolist()
    y_test = test['label'].tolist()

    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1,2), max_features=int(max_features), stop_words='english')),
        ('clf', LogisticRegression(C=float(C), max_iter=200, solver='lbfgs'))
    ])

    print('Fitting pipeline on train+val (combined)...')
    X_comb = X_train + X_val
    y_comb = y_train + y_val
    pipe.fit(X_comb, y_comb)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', acc)
    print(classification_report(y_test, y_pred))

    outp = PROJECT / out_path
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, str(outp))
    print('Saved final model to', outp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/IMDB_Dataset.csv')
    parser.add_argument('--out', type=str, default='models/model_final_imdb.joblib')
    parser.add_argument('--max-features', type=int, default=20000)
    parser.add_argument('--C', type=float, default=1.0)
    args = parser.parse_args()
    main(data_path=args.data, out_path=args.out, max_features=args.max_features, C=args.C)
