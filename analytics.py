import pandas as pd
import numpy as np

def audit_and_clean_data(df, auto_clean=True):
    """
    Evaluates data health (missing cells, duplicates) and optionally cleans it.
    """
    raw_rows = len(df)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    total_cells = df.shape[0] * df.shape[1]
    
    health_score = round(max(0, 100 - ((missing_cells + duplicate_rows * df.shape[1]) / max(1, total_cells)) * 100), 1)
    
    cleaned_df = df.copy()
    if auto_clean:
        cleaned_df = cleaned_df.drop_duplicates()
        for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].fillna("Unknown")
            
    return cleaned_df, {
        "raw_rows": raw_rows,
        "clean_rows": len(cleaned_df),
        "columns": df.shape[1],
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "health_score": health_score
    }

def find_anomalies(df, column_name):
    """
    Uses Interquartile Range (IQR) and Z-Scores to catch statistical outliers.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    anomalies = df[(df[column_name] < lower_limit) | (df[column_name] > upper_limit)].copy()
    
    mean = df[column_name].mean()
    std_dev = df[column_name].std()
    
    if std_dev != 0:
        anomalies['Z_Score'] = ((anomalies[column_name] - mean) / std_dev).round(2)
    else:
        anomalies['Z_Score'] = 0.0
        
    return anomalies

def calculate_profit_leak(df, metric_col):
    """
    Calculates exact money/value lost on days that fell below the business's normal median baseline.
    """
    median_val = df[metric_col].median()
    underperforming_days = df[df[metric_col] < median_val]
    # Total gap below normal median performance
    total_leak = (median_val - underperforming_days[metric_col]).sum()
    return round(max(0, total_leak), 2), len(underperforming_days)

def full_statistical_profile(df, metric_col):
    """
    Computes distribution metrics, period growth, and correlation drivers.
    """
    series = df[metric_col].dropna()
    mean_val = series.mean()
    median_val = series.median()
    std_val = series.std()
    cv = round((std_val / mean_val) * 100, 1) if mean_val != 0 else 0
    skew_val = round(series.skew(), 2) if len(series) > 2 else 0.0
    
    mid = len(series) // 2
    first_half = series.iloc[:mid].mean() if mid > 0 else mean_val
    second_half = series.iloc[mid:].mean() if mid > 0 else mean_val
    growth_pct = round(((second_half - first_half) / first_half) * 100, 1) if first_half != 0 else 0.0
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    drivers = pd.DataFrame()
    top_driver = None
    top_corr = 0.0
    slope = 0.0
    
    if len(numeric_df.columns) > 1:
        corrs = numeric_df.corr()[metric_col].drop(metric_col).dropna()
        if not corrs.empty:
            drivers = corrs.reset_index()
            drivers.columns = ['Feature', 'Correlation']
            drivers['Abs_Corr'] = drivers['Correlation'].abs()
            drivers = drivers.sort_values('Abs_Corr', ascending=False)
            top_driver = drivers.iloc[0]['Feature']
            top_corr = round(drivers.iloc[0]['Correlation'], 2)
            
            # Linear slope: how much target metric rises per 1 unit increase in top_driver
            if df[top_driver].std() != 0:
                slope = round(np.polyfit(df[top_driver], df[metric_col], 1)[0], 2)

    return {
        "total": round(series.sum(), 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "min": round(series.min(), 2),
        "max": round(series.max(), 2),
        "cv": cv,
        "skew": skew_val,
        "growth_pct": growth_pct,
        "top_driver": top_driver,
        "top_corr": top_corr,
        "driver_slope": slope,
        "drivers_df": drivers
    }

def analyze_day_patterns(df, metric_col):
    """
    Extracts Day-of-Week performance and Weekday vs. Weekend lift.
    """
    temp_df = df.copy()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    found_day_col = False

    for col in temp_df.columns:
        if temp_df[col].dtype == 'object':
            sample_vals = temp_df[col].dropna().astype(str).str.strip().str.title().unique()
            if any(d in day_order for d in sample_vals):
                temp_df['Day_Name'] = temp_df[col].astype(str).str.strip().str.title()
                found_day_col = True
                break

    if not found_day_col:
        for col in temp_df.columns:
            if 'date' in col.lower() or temp_df[col].dtype == 'object':
                try:
                    parsed_dates = pd.to_datetime(temp_df[col], errors='coerce')
                    if parsed_dates.notna().sum() >= len(temp_df) * 0.7:
                        temp_df['Day_Name'] = parsed_dates.dt.day_name()
                        found_day_col = True
                        break
                except Exception:
                    continue

    if not found_day_col:
        return None

    day_summary = temp_df.groupby('Day_Name')[metric_col].agg(
        Average='mean',
        Total='sum',
        Peak='max',
        Days_Count='count'
    ).reset_index()

    day_summary['Day_Name'] = pd.Categorical(day_summary['Day_Name'], categories=day_order, ordered=True)
    day_summary = day_summary.sort_values('Day_Name').dropna(subset=['Day_Name'])
    day_summary['Average'] = day_summary['Average'].round(2)

    temp_df['Day_Type'] = temp_df['Day_Name'].apply(
        lambda d: 'Weekend (Sat-Sun)' if d in ['Saturday', 'Sunday'] else 'Weekday (Mon-Fri)'
    )
    type_summary = temp_df.groupby('Day_Type')[metric_col].mean().round(2).to_dict()

    return {
        "by_day": day_summary,
        "by_type": type_summary
    }

def analyze_categories(df, metric_col):
    """
    Finds categorical columns (e.g., Product, Region, Category) and calculates segment share.
    """
    day_words = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    cat_results = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        unique_vals = df[col].dropna().unique()
        if 2 <= len(unique_vals) <= 15 and 'date' not in col.lower():
            lower_vals = {str(v).strip().lower() for v in unique_vals}
            if not lower_vals.intersection(day_words):
                seg = df.groupby(col)[metric_col].agg(Total='sum', Average='mean', Count='count').reset_index()
                seg = seg.sort_values('Total', ascending=False)
                seg['Share_%'] = round((seg['Total'] / seg['Total'].sum()) * 100, 1)
                cat_results[col] = seg
                
    return cat_results
