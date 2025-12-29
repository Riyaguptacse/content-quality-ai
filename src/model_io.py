from pathlib import Path
import joblib

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def save_artifacts(obj, filename: str) -> str:
    """
    Save a Python object (model/pipeline) into the models/ folder.
    """
    path = MODEL_DIR / filename
    joblib.dump(obj, path)
    return str(path)

def load_artifacts(filename: str):
    """
    Load a saved object from the models/ folder.
    """
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Train first (python src/train.py).")
    return joblib.load(path)
