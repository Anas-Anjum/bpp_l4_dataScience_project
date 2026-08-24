import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found at {file_path}")
    
    logging.info(f"Ingesting raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Extraction successful: {df.shape[0]:,} rows loaded.")
    return df
