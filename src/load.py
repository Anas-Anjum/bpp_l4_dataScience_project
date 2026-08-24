import pandas as pd
import sqlite3
import logging

def load_to_parquet(df: pd.DataFrame, output_path: str):
    logging.info(f"Writing Parquet file to: {output_path}")
    df.to_parquet(output_path, index=False)

def load_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str = "cleaned_transactions"):
    logging.info(f"Loading data into SQLite DB: {db_path}")
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    logging.info("Database load operation completed successfully.")
