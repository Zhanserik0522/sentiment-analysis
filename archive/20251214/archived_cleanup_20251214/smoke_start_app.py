import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def import_app_module():
    # Import app.app as a module without running Streamlit UI
    spec = importlib.util.spec_from_file_location('app.app', str(ROOT / 'app' / 'app.py'))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        # app.py may call sys.exit in CLI mode; ignore for headless import
        pass
    return mod


def main():
    print('Importing app module...')
    mod = import_app_module()
    print('Imported:', mod)
    # Try to call the loader function if available
    loader = getattr(mod, 'load_model_from_path', None)
    if loader is None:
        print('load_model_from_path not found in app module — abort')
        raise SystemExit(2)
    print('Loading model (default candidates)...')
    mdl, path = loader()
    print('Loaded model path:', path)
    print('Model type:', type(mdl))
    classes = getattr(mdl, 'classes_', None)
    print('Classes:', classes)
    # Try a sample prediction
    text = 'I hate it'
    print('Running a sample prediction for:', text)
    # attempt to import clean_text helper
    try:
        from src.preprocessing import clean_text
        cleaned = clean_text(text)
    except Exception:
        cleaned = text
    try:
        pred = mdl.predict([cleaned])[0]
        print('Predicted:', pred)
    except Exception as e:
        print('Prediction failed:', e)


if __name__ == '__main__':
    main()
