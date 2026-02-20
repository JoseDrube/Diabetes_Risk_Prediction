from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/diabetes.csv")
INTERIM_DATA_PATH = Path("data/interim/diabetes.csv")
PROCESSED_DATA_PATH = Path("data/processed/diabetes.csv")

def load_raw_data() -> pd.DataFrame:
    """Load raw diabetes dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"{RAW_DATA_PATH} not found.")
    
    return pd.read_csv(RAW_DATA_PATH)