import pandas as pd
import numpy as np

def engineer_sales_features(
    tickets_upto_t: pd.DataFrame,
    df_merged: pd.DataFrame,
    lag_days_sales: int,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert sales-gebaseerde features.
    
    Parameters:
        tickets_upto_t: Tickets tot aan tijdstip T
        df_merged: Gemergde DataFrame tot nu toe
        lag_days_sales: Aantal dagen voor lag features
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        DataFrame met sales features toegevoegd
    """
    if verbose:
        print("[engineer_sales_features] Stap 4: Sales Features...")
    
    # Start met events in df_merged
    sales_agg = pd.DataFrame({'event_name': df_merged['event_name'].unique()})
    
    if not tickets_upto_t.empty:
        # Cumulative sales
        df_cum = tickets_upto_t.groupby('event_name').size().reset_index(name='cumulative_sales_at_t')
        sales_agg = pd.merge(sales_agg, df_cum, on='event_name', how='left')
        sales_agg['cumulative_sales_at_t'] = sales_agg['cumulative_sales_at_t'].fillna(0)
        sales_agg['log_cumulative_sales_at_t'] = np.log1p(sales_agg['cumulative_sales_at_t'])
        
        # Average daily sales velocity
        avg_daily = tickets_upto_t.groupby('event_name')['days_until_event'].agg(
            lambda x: len(x) / (x.max() - x.min() + 1) if x.nunique() > 1 else len(x)
        ).reset_index(name='avg_daily_sales_before_t')
        sales_agg = pd.merge(sales_agg, avg_daily, on='event_name', how='left')
        sales_agg['log_avg_daily_sales_before_t'] = np.log1p(sales_agg['avg_daily_sales_before_t'].fillna(0))
        
        # Sales acceleration
        daily_sales = tickets_upto_t.groupby(['event_name', 'days_until_event']).size().reset_index(name='daily_sales')
        if not daily_sales.empty:
            daily_sales = daily_sales.sort_values(['event_name', 'days_until_event'])
            daily_sales['acceleration'] = daily_sales.groupby('event_name')['daily_sales'].diff().fillna(0)
            accel = daily_sales.groupby('event_name')['acceleration'].mean().reset_index(name='avg_acceleration')
            sales_agg = pd.merge(sales_agg, accel, on='event_name', how='left')
    
    # Lijst van sales kolommen die mogelijk missen
    cols_to_default_sales = [
        'cumulative_sales_at_t', 'log_cumulative_sales_at_t',
        'avg_daily_sales_before_t', 'log_avg_daily_sales_before_t',
        'avg_acceleration'
    ]
    
    # Zorg dat alle kolommen bestaan en geen NaN waarden hebben
    for col in cols_to_default_sales:
        if col not in sales_agg.columns:
            sales_agg[col] = 0  # Voeg kolom toe met 0 als hij mist
            if verbose:
                print(f"Default 0 toegevoegd voor missende sales feature: {col}")
        elif sales_agg[col].isnull().any():  # Als kolom bestaat, vul eventuele NaNs
            sales_agg[col] = sales_agg[col].fillna(0)
    
    # Merge sales features met df_merged
    df_merged = pd.merge(df_merged, sales_agg, on='event_name', how='left')
    
    return df_merged 