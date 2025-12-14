import joblib
from pathlib import Path
p=Path('models')
print('Models dir exists:', p.exists())
for f in p.glob('*.joblib'):
    print('Found:', f.name)
    try:
        m=joblib.load(f)
        print('  type:', type(m))
        if hasattr(m, 'named_steps'):
            clf = m.named_steps.get('clf')
        else:
            clf = m
        if hasattr(clf, 'classes_'):
            print('  classes_:', clf.classes_)
        else:
            print('  no classes_ attribute')
    except Exception as e:
        print('  load error', e)
