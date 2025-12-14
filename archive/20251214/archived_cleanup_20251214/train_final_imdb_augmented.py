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
import random


SHORT_POS = [
    'good', 'great', 'love', 'like', 'excellent', 'awesome', 'nice', 'fantastic', 'amazing', 'perfect'
]
SHORT_NEG = [
    'bad', 'terrible', 'hate', 'awful', 'worst', 'poor', 'disappointing', 'horrible', 'trash', 'sad'
]


def make_short_examples(n_per_class=2000):
    examples = []
    for _ in range(n_per_class):
        w = random.choice(SHORT_POS)
        examples.append((w, 'positive'))
    for _ in range(n_per_class):
        w = random.choice(SHORT_NEG)
        examples.append((w, 'negative'))
    return examples


def main(data_path='data/IMDB_Dataset.csv', out_path='models/model_final_imdb_augmented.joblib', max_features=20000, C=1.0):
    DATA = PROJECT / data_path
    if not DATA.exists():
        raise SystemExit(f'IMDB dataset not found at {DATA}.')
    print('Loading IMDB dataset...')
    df = load_dataset(str(DATA))
    if 'review' in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
    df = preprocess_df(df)
    train, val, test = split_data(df, test_size=0.1, val_size=0.1, random_state=42)
    print(f'Train/Val/Test sizes before augmentation: {len(train)}/{len(val)}/{len(test)}')

    # create short examples and upsample into train
    short_ex = make_short_examples(n_per_class=3000)
    short_df = []
    for t, lbl in short_ex:
        short_df.append({'text': t, 'label': lbl})
    import pandas as pd
    short_df = pd.DataFrame(short_df)
    # prepend short examples to train
    train_aug = pd.concat([pd.DataFrame(train), short_df], ignore_index=True)
    print('Train size after augmentation:', len(train_aug))

    X_comb = list(train_aug['text']) + list(val['text'])
    y_comb = list(train_aug['label']) + list(val['label'])

    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1,2), max_features=int(max_features), stop_words='english')),
        ('clf', LogisticRegression(C=float(C), max_iter=300, solver='lbfgs'))
    ])

    print('Fitting augmented pipeline on train+val...')
    pipe.fit(X_comb, y_comb)

    y_pred = pipe.predict(list(test['text']))
    acc = accuracy_score(list(test['label']), y_pred)
    print('Test accuracy after augmentation:', acc)
    print(classification_report(list(test['label']), y_pred))

    outp = PROJECT / out_path
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, str(outp))
    print('Saved augmented model to', outp)

    # print control predictions
    controls = ['I like it', 'like', 'I hate it', 'hate', 'good', 'bad']
    for c in controls:
        try:
            print('INPUT:', c)
            pred = pipe.predict([c])[0]
            probs = pipe.predict_proba([c])[0]
            print('PRED:', pred, 'PROBS:', dict(zip(pipe.classes_, probs)))
        except Exception as e:
            print('control pred failed:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/IMDB_Dataset.csv')
    parser.add_argument('--out', type=str, default='models/model_final_imdb_augmented.joblib')
    parser.add_argument('--max-features', type=int, default=20000)
    parser.add_argument('--C', type=float, default=1.0)
    args = parser.parse_args()
    main(data_path=args.data, out_path=args.out, max_features=args.max_features, C=args.C)
