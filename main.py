from src.extract import extract_data
from src.transform import transform_data
from src.load import load_to_parquet, load_to_sqlite

def run_pipeline():
    print("--- Starting ETL Pipeline Execution ---")
    
    raw_path = "data/raw/creditcard.csv"
    parquet_path = "data/processed/creditcard_clean.parquet"
    db_path = "data/processed/fraud_warehouse.db"
    
    raw_df = extract_data(raw_path)
    clean_df = transform_data(raw_df)
    
    load_to_parquet(clean_df, parquet_path)
    load_to_sqlite(clean_df, db_path)
    
    print("\n[SUCCESS] Pipeline executed successfully!")

if __name__ == "__main__":
    run_pipeline()
