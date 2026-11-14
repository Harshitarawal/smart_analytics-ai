import pandas as pd
import numpy as np
import scipy.stats as stats

def analyze_inventory_risk(df, weather_boost_pct=0):
    """
    Calculates Poisson Stockout Probability, Safety Stock, and Economic Order Quantity (EOQ)
    for every SKU in the inventory sheet.
    """
    inv_df = df.copy()
    
    # Adjust daily demand velocity if user simulates a weekend/weather surge
    demand_multiplier = 1 + (weather_boost_pct / 100.0)
    inv_df['Adjusted_Daily_Sales'] = (inv_df['Daily_Sales_Avg'] * demand_multiplier).round(2)
    
    # 1. Expected demand while waiting for supplier delivery (Lead Time)
    inv_df['Lead_Time_Demand'] = inv_df['Adjusted_Daily_Sales'] * inv_df['Lead_Time_Days']
    
    # 2. Poisson Cumulative Distribution Function (CDF) for Stockout Probability
    stockout_probs = []
    for _, row in inv_df.iterrows():
        mu = max(0.1, row['Lead_Time_Demand'])
        current_stock = int(row['Current_Stock'])
        prob_safe = stats.poisson.cdf(current_stock, mu=mu)
        prob_out = round((1 - prob_safe) * 100, 1)
        stockout_probs.append(prob_out)
        
    inv_df['Stockout_Risk_%'] = stockout_probs
    
    # 3. Statistical Safety Stock (95% service level, Z = 1.645)
    # Assumes daily demand std dev is ~25% of average daily sales
    inv_df['Safety_Stock'] = np.ceil(1.645 * (0.25 * inv_df['Adjusted_Daily_Sales']) * np.sqrt(inv_df['Lead_Time_Days'])).astype(int)
    
    # 4. Reorder Point = Lead Time Demand + Safety Stock
    inv_df['Reorder_Point'] = np.ceil(inv_df['Lead_Time_Demand'] + inv_df['Safety_Stock']).astype(int)
    
    # 5. Economic Order Quantity (EOQ) = sqrt((2 * Annual_Demand * Order_Cost) / Holding_Cost)
    annual_demand = inv_df['Adjusted_Daily_Sales'] * 365
    order_cost = inv_df['Order_Cost'] if 'Order_Cost' in inv_df.columns else 250
    holding_cost = inv_df['Holding_Cost'] if 'Holding_Cost' in inv_df.columns else 15
    inv_df['Optimal_EOQ'] = np.ceil(np.sqrt((2 * annual_demand * order_cost) / np.maximum(1, holding_cost))).astype(int)
    
    # 6. Days of Stock Remaining
    inv_df['Days_Left'] = np.where(
        inv_df['Adjusted_Daily_Sales'] > 0,
        (inv_df['Current_Stock'] / inv_df['Adjusted_Daily_Sales']).round(1),
        999
    )
    
    # 7. Assign Traffic-Light Status
    def get_status(row):
        if row['Stockout_Risk_%'] >= 70 or row['Current_Stock'] <= row['Reorder_Point']:
            return "🔴 CRITICAL REORDER"
        elif row['Stockout_Risk_%'] >= 30:
            return "🟡 WATCH CLOSELY"
        else:
            return "🟢 SAFE STOCK"
            
    inv_df['Status'] = inv_df.apply(get_status, axis=1)
    return inv_df
