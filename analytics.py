import pandas as pd
import numpy as np

def find_anomalies(df, column_name):
    """
    Uses Interquartile Range (IQR) and Z-Scores to find unusual spikes or drops in data.
    """
    # 1. Calculate the 25th percentile (Q1) and 75th percentile (Q3)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # 2. Define the normal boundaries
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    # 3. Filter out the rows that fall outside normal boundaries
    anomalies = df[(df[column_name] < lower_limit) | (df[column_name] > upper_limit)].copy()
    
    # 4. Calculate Z-Score to measure how extreme each anomaly is
    mean = df[column_name].mean()
    std_dev = df[column_name].std()
    
    if std_dev != 0:
        anomalies['Z_Score'] = ((anomalies[column_name] - mean) / std_dev).round(2)
    else:
        anomalies['Z_Score'] = 0.0
        
    return anomalies
