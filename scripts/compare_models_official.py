from pathlib import Path
import sys
import joblib
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import load_dataset, preprocess_df, split_data
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


def eval_model_on_test(model_path, X_test, y_test):
    m = joblib.load(model_path)
    # assume pipeline expecting raw text
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'model': Path(model_path).name,
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'report': report
    }


def main(data_path='data/IMDB_Dataset.csv'):
    DATA = ROOT / data_path
    if not DATA.exists():
        raise SystemExit('Dataset not found: ' + str(DATA))

    df = load_dataset(str(DATA))
    if 'review' in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
    df = preprocess_df(df)
    train, val, test = split_data(df, test_size=0.1, val_size=0.1, random_state=42)
    X_test = test['text'].tolist()
    y_test = test['label'].tolist()

    models = [ROOT / 'models' / 'lr_best.joblib', ROOT / 'models' / 'rf_best.joblib']
    results = []
    for m in models:
        if not m.exists():
            print('Model not found, skipping:', m)
            continue
        print('Evaluating', m.name)
        res = eval_model_on_test(str(m), X_test, y_test)
        results.append(res)

    out_csv = ROOT / 'models' / 'MODEL_COMPARISON_OFFICIAL.csv'
    with open(out_csv, 'w', encoding='utf-8', newline='') as cf:
        fieldnames = ['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    out_md = ROOT / 'models' / 'MODEL_COMPARISON_OFFICIAL.md'
    with open(out_md, 'w', encoding='utf-8') as mf:
        mf.write('# Model comparison (official)\n\n')
        for r in results:
            mf.write(f"## {r['model']}\n\n")
            mf.write(f"- accuracy: {r['accuracy']:.4f}\n")
            mf.write(f"- precision_macro: {r['precision_macro']:.4f}\n")
            mf.write(f"- recall_macro: {r['recall_macro']:.4f}\n")
            mf.write(f"- f1_macro: {r['f1_macro']:.4f}\n\n")
            mf.write('### classification_report\n')
            mf.write(json.dumps(r['report'], indent=2))
            mf.write('\n\n')

    print('Wrote CSV to', out_csv)
    print('Wrote MD report to', out_md)


if __name__ == '__main__':
    main()
