import pandas as pd
from sklearn.preprocessing import RobustScaler
import logging

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting data transformation...")
    df_clean = df.copy()
    
    scaler = RobustScaler()
    df_clean['scaled_amount'] = scaler.fit_transform(df_clean[['Amount']])
    df_clean['scaled_time'] = scaler.fit_transform(df_clean[['Time']])
    
    df_clean['hour_of_day'] = (df_clean['Time'] // 3600) % 24
    df_clean.drop(columns=['Amount', 'Time'], inplace=True)
    
    logging.info("Transformation completed successfully.")
    return df_clean
