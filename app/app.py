import sys
import pathlib
try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    joblib = None  # type: ignore
    _HAS_JOBLIB = False
import pickle
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    # Streamlit not available in PATH / environment. We'll fall back to a
    # minimal CLI mode so the app can still be used for quick smoke tests.
    st = None  # type: ignore
    _HAS_STREAMLIT = False

# Ensure project root is on sys.path so `src` package imports work when
# running the app via `streamlit run` from any CWD.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

if _HAS_STREAMLIT:
    st.title('Sentiment Analysis Demo')

def load_model_from_path(path: str | None = None):
    """Load a joblib model from path or fallback candidates. Cached to avoid reloads."""
    candidates = [
        # prefer the canonical final model
        Path('models') / 'model_final.joblib',
        Path('models') / 'model_final_imdb_augmented.joblib',
        Path('models') / 'model_grid_best_calibrated.joblib',
    ]
    if path:
        try:
            p = Path(path)
            if p.exists():
                return _load_artifact(p), str(p)
        except Exception:
            pass
    for p in candidates:
        try:
            if p.exists():
                return _load_artifact(p), str(p)
        except Exception:
            continue
    return None, None


def _load_artifact(p: Path):
    """Load a model artifact using joblib when available, otherwise fall
    back to pickle. Returns the loaded object or raises the error from the
    attempted loader."""
    if _HAS_JOBLIB and joblib is not None:
        try:
            return joblib.load(p)
        except Exception:
            # fall through to pickle
            pass
    # Try pickle as a fallback
    with open(p, 'rb') as fh:
        return pickle.load(fh)

clf = None
clf_path = None
if _HAS_STREAMLIT:
    # If the user has previously selected a model in this session, prefer that
    if 'selected_model' in st.session_state:
        clf, clf_path = load_model_from_path(st.session_state.get('selected_model'))
    else:
        clf, clf_path = load_model_from_path()
else:
    # In CLI mode we still want to load the preferred candidate model
    clf, clf_path = load_model_from_path()

if _HAS_STREAMLIT:
    st.sidebar.header('Model selection')

# Heuristic toggle: allow forcing negative on presence of strong negative words
HEURISTIC_NEG_WORDS = {'hate', 'awful', 'worst', 'terrible', 'disgusting', 'horrible'}
def apply_negative_heuristic(text: str):
    t = text.lower()
    for w in HEURISTIC_NEG_WORDS:
        if w in t:
            return True
    return False

# safe rerun helper: some Streamlit builds may not expose experimental_rerun
def safe_rerun():
    # Some Streamlit builds don't expose experimental_rerun; use getattr to avoid
    # AttributeError when referencing the attribute.
    fn = getattr(st, 'experimental_rerun', None)
    if callable(fn):
        try:
            fn()
        except Exception:
            st.info('Please refresh the page to apply the change (auto-rerun failed).')
            st.stop()
    else:
        st.info('Please refresh the page to apply the change (auto-rerun not supported in this Streamlit build).')
        st.stop()

# Discover available models for selector (models/ and models/full_search/)
def discover_models():
    base = Path('models')
    files = []
    if base.exists():
        files += list(base.glob('*.joblib'))
    full = base / 'full_search'
    if full.exists():
        files += list(full.glob('*.joblib'))
    # sort by name
    files = sorted(files, key=lambda p: p.name)
    return files

available = discover_models()
available_names = [str(p) for p in available]
selected = None
selected_model_path = None

# Build friendly labels and descriptions
def friendly_label(p: Path) -> str:
    # build a compact but unique label: name | size_kb | mtime
    try:
        stat = p.stat()
        size_kb = stat.st_size // 1024
        mtime = int(stat.st_mtime)
        return f"{p.name} — {size_kb}KB — {mtime}"
    except Exception:
        return p.name

def model_description(p: Path) -> str:
    try:
        m = joblib.load(p)
        classes = getattr(m, 'classes_', None)
        if classes is None:
            cls_text = 'unknown'
        else:
            cls_text = ','.join(list(classes))
        kind = 'binary' if (classes is not None and len(classes) == 2) else 'multi-class'
        location = 'full_search' if 'full_search' in str(p) else 'models'
        stat = p.stat()
        size_kb = stat.st_size // 1024
        mtime = stat.st_mtime
        return (
            f'filename: {p.name}\n'
            f'location: {location}\n'
            f'classes: {cls_text}\n'
            f'type: {kind}\n'
            f'size_kb: {size_kb}\n'
            f'mtime: {mtime}'
        )
    except Exception as e:
        return f'failed to load ({e})'

labels = [friendly_label(p) for p in available]
descriptions = {friendly_label(p): model_description(p) for p in available}

if _HAS_STREAMLIT and labels:
    # let selectbox return Path objects but show friendly labels
    sel_path = st.sidebar.selectbox('Choose model artifact', available, format_func=friendly_label)
    selected = str(sel_path)
    st.sidebar.write(model_description(sel_path))
    # Auto-load on selection change: set session state. If Streamlit supports
    # experimental_rerun, use it; otherwise load the model immediately in-place
    prev = st.session_state.get('selected_model')
    if prev != selected:
        st.session_state['selected_model'] = selected
        if getattr(st, 'experimental_rerun', None) and callable(getattr(st, 'experimental_rerun')):
            safe_rerun()
        else:
            # load model now so the UI continues without requiring a manual refresh
            try:
                clf, clf_path = load_model_from_path(selected)
            except Exception:
                clf, clf_path = None, None
    # keep explicit load button for compatibility / manual control
    if st.sidebar.button('Load selected model'):
        st.session_state['selected_model'] = selected
        if getattr(st, 'experimental_rerun', None) and callable(getattr(st, 'experimental_rerun')):
            safe_rerun()
        else:
            try:
                clf, clf_path = load_model_from_path(selected)
            except Exception:
                clf, clf_path = None, None

    # minimal UI: selection only
elif _HAS_STREAMLIT:
    st.sidebar.info('No .joblib models found in models/ or models/full_search/')

# Debug: show session selection and loaded model path
if _HAS_STREAMLIT:
    st.sidebar.markdown('---')
    # heuristic toggle in the sidebar
    if 'enable_neg_heuristic' not in st.session_state:
        st.session_state['enable_neg_heuristic'] = False
    st.session_state['enable_neg_heuristic'] = st.sidebar.checkbox('Enable negative-word heuristic', value=st.session_state['enable_neg_heuristic'])
    st.sidebar.write('session.selected_model = ' + str(st.session_state.get('selected_model')))
    st.sidebar.write('loaded model path = ' + str(clf_path))

# Show model hash and top features on demand
# compute_model_info removed per user request; model metadata display was deleted.


# Sidebar model-info button and display removed per user request.

def _cli_predict_loop(clf, clf_path):
    """Simple CLI loop to allow using the sentiment model when Streamlit is
    not available. Type 'quit' or Ctrl-C to exit."""
    try:
        from src.preprocessing import clean_text as _clean
    except Exception:
        # Fallback simple cleaner if package deps are missing
        def _clean(s: str) -> str:
            return ' '.join([w.lower() for w in s.split() if w.isalnum() or w.isalpha()])
    print('\nSentiment CLI mode — model loaded from:', clf_path)
    print("Enter text to analyze (type 'quit' to exit):")
    try:
        while True:
            text = input('> ').strip()
            if not text:
                continue
            if text.lower() in {'quit', 'exit'}:
                print('Exiting.')
                break
            cleaned = _clean(text)
            print('cleaned text:', cleaned)
            try:
                pred = clf.predict([cleaned])[0]
                print('Predicted sentiment:', pred)
            except Exception as e:
                print('Prediction failed:', e)
            try:
                if hasattr(clf, 'predict_proba'):
                    probs = clf.predict_proba([cleaned])[0]
                    cls = getattr(clf, 'classes_', [])
                    probs_display = {str(c): float(p) for c, p in zip(cls, probs)}
                    print('Prediction probabilities:', probs_display)
            except Exception:
                pass
            print('---')
    except KeyboardInterrupt:
        print('\nInterrupted — exiting.')


if not _HAS_STREAMLIT:
    # Run CLI fallback when streamlit isn't installed or the command is not
    # available. This helps with quick debugging and makes the app "work"
    # even without Streamlit in PATH.
    if clf is None:
        print('No model found. Checked default candidates in models/.')
    else:
        _cli_predict_loop(clf, clf_path)

if _HAS_STREAMLIT:
    # Restore the main input area and prediction UI for the Streamlit app.
    # Use lazy import for preprocessing to avoid pulling heavy deps at module import.
    try:
        from src.preprocessing import clean_text as _clean_text
    except Exception:
        def _clean_text(s: str) -> str:
            return ' '.join([w.lower() for w in s.split() if w.isalnum() or w.isalpha()])

    text = st.text_area('Enter text to analyze')

    if st.button('Predict'):
        if not text or not text.strip():
            st.warning('Please enter some text')
        elif clf is None:
            st.error('Model not found. Train models first or place model file at models/best_model.joblib')
        else:
            cleaned = _clean_text(text)
            # apply heuristic before model prediction if enabled
            forced_pred = None
            if st.session_state.get('enable_neg_heuristic') and apply_negative_heuristic(text):
                st.caption('Negative heuristic matched — forcing negative')
                forced_pred = 'negative'
                try:
                    if hasattr(clf, 'predict_proba'):
                        cls = getattr(clf, 'classes_', [])
                        probs_display = {str(c): (1.0 if str(c) == 'negative' else 0.0) for c in cls}
                        st.write('Prediction probabilities:', probs_display)
                except Exception:
                    pass
            st.caption(f'cleaned text: {cleaned}')
            try:
                if forced_pred is not None:
                    st.write('Predicted sentiment:', forced_pred)
                else:
                    pred = clf.predict([cleaned])[0]
                    st.write('Predicted sentiment:', pred)
            except Exception as e:
                st.error(f'Prediction failed: {e}')
            try:
                if hasattr(clf, 'predict_proba'):
                    probs = clf.predict_proba([cleaned])[0]
                    cls = getattr(clf, 'classes_', [])
                    probs_display = {str(c): float(p) for c, p in zip(cls, probs)}
                    st.write('Prediction probabilities:', probs_display)
            except Exception:
                pass
            try:
                classes = getattr(clf, 'classes_', None)
                classes_list = list(classes) if classes is not None else 'unknown'
                st.caption(f'Using model: {clf_path} | classes: {classes_list}')
            except Exception:
                pass
