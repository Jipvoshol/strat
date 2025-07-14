import pandas as pd
import numpy as np
from typing import Tuple

def engineer_time_features(
    tickets: pd.DataFrame, 
    tickets_upto_t: pd.DataFrame,
    df_total: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert tijd-gebaseerde features.
    
    Parameters:
        tickets: Hoofd tickets DataFrame
        tickets_upto_t: Tickets tot aan tijdstip T
        df_total: Basis aggregatie DataFrame
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        DataFrame met tijd features
    """
    if verbose:
        print("[engineer_time_features] Stap 3: Tijd Features...")
    
    event_time_agg = tickets.groupby('event_name').agg(
        first_event_date=('first_event_date_start', 'first'), 
        last_event_date=('last_event_date_end', 'first'),
        first_sale_date=('verkoopdatum', 'min'), 
        last_sale_date_overall=('verkoopdatum', 'max')
    ).reset_index()

    # Controleer eerst of de datumkolommen daadwerkelijk datetime objecten zijn
    for date_col in ['first_event_date', 'last_event_date', 'first_sale_date', 'last_sale_date_overall']:
        if date_col in event_time_agg.columns:
            if not pd.api.types.is_datetime64_dtype(event_time_agg[date_col]):
                event_time_agg[date_col] = pd.to_datetime(event_time_agg[date_col], errors='coerce')

    # Filter ontbrekende datum waarden weg
    for date_col in ['first_event_date', 'last_event_date', 'first_sale_date', 'last_sale_date_overall']:
        if date_col in event_time_agg.columns:
            missing_dates = event_time_agg[date_col].isna().sum()
            if missing_dates > 0:
                before_count = len(event_time_agg)
                event_time_agg = event_time_agg[event_time_agg[date_col].notna()]
                after_count = len(event_time_agg)
                if verbose:
                    print(f"{missing_dates} rijen met ontbrekende {date_col} verwijderd. Rijen over: {after_count}/{before_count}")

    last_sales_t = tickets_upto_t.groupby('event_name')['verkoopdatum'].max().reset_index(name='last_sale_date_upto_t')
    event_time_agg = pd.merge(event_time_agg, last_sales_t, on='event_name', how='left')

    # Zorg ervoor dat last_sale_date_upto_t ook een datetime is
    if 'last_sale_date_upto_t' in event_time_agg.columns:
        if not pd.api.types.is_datetime64_dtype(event_time_agg['last_sale_date_upto_t']):
            event_time_agg['last_sale_date_upto_t'] = pd.to_datetime(event_time_agg['last_sale_date_upto_t'], errors='coerce')
        
        missing_dates = event_time_agg['last_sale_date_upto_t'].isna().sum()
        if missing_dates > 0:
            before_count = len(event_time_agg)
            event_time_agg = event_time_agg[event_time_agg['last_sale_date_upto_t'].notna()]
            after_count = len(event_time_agg)
            if verbose:
                print(f"{missing_dates} rijen met ontbrekende last_sale_date_upto_t verwijderd. Rijen over: {after_count}/{before_count}")

    # Event duration
    valid_event_dates = event_time_agg['first_event_date'].notna() & event_time_agg['last_event_date'].notna()
    event_time_agg['event_duration_hours'] = np.nan
    event_time_agg.loc[valid_event_dates, 'event_duration_hours'] = (
        event_time_agg.loc[valid_event_dates, 'last_event_date'] - 
        event_time_agg.loc[valid_event_dates, 'first_event_date']
    ).dt.total_seconds() / 3600
    
    # Vervang onrealistische waarden
    event_time_agg.loc[
        (event_time_agg['event_duration_hours'] < 0) | 
        (event_time_agg['event_duration_hours'] > 30*24), 
        'event_duration_hours'
    ] = np.nan
    
    median_event_duration = event_time_agg['event_duration_hours'].median()
    event_time_agg['event_duration_hours'] = event_time_agg['event_duration_hours'].fillna(
        median_event_duration if pd.notna(median_event_duration) else 4
    )

    # Sales duration
    sales_dur_col = 'sales_duration_hours_upto_t'
    log_sales_dur_col = f'log_{sales_dur_col}'
    valid_sale_dates = event_time_agg['first_sale_date'].notna() & event_time_agg['last_sale_date_upto_t'].notna()
    event_time_agg[sales_dur_col] = np.nan
    event_time_agg.loc[valid_sale_dates, sales_dur_col] = (
        event_time_agg.loc[valid_sale_dates, 'last_sale_date_upto_t'] - 
        event_time_agg.loc[valid_sale_dates, 'first_sale_date']
    ).dt.total_seconds() / 3600
    
    event_time_agg.loc[event_time_agg[sales_dur_col] < 0, sales_dur_col] = 0
    event_time_agg[sales_dur_col] = event_time_agg[sales_dur_col].fillna(0)
    event_time_agg[log_sales_dur_col] = np.log1p(event_time_agg[sales_dur_col])

    # Cyclical time features
    event_time_agg['day_of_week'] = event_time_agg['first_event_date'].dt.dayofweek.fillna(-1).astype(int)
    event_time_agg['event_month'] = event_time_agg['first_event_date'].dt.month.fillna(-1).astype(int)
    event_time_agg['event_year'] = event_time_agg['first_event_date'].dt.year.fillna(-1).astype(int)
    event_time_agg['is_weekend'] = event_time_agg['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding
    event_time_agg['month_sin'] = np.sin(2 * np.pi * event_time_agg['event_month']/12)
    event_time_agg['month_cos'] = np.cos(2 * np.pi * event_time_agg['event_month']/12)
    event_time_agg['day_of_week_sin'] = np.sin(2 * np.pi * event_time_agg['day_of_week']/7)
    event_time_agg['day_of_week_cos'] = np.cos(2 * np.pi * event_time_agg['day_of_week']/7)

    # Drop kolommen die we niet meer nodig hebben
    cols_to_drop_from_time_agg = [
        'last_event_date', 'first_sale_date', 
        'last_sale_date_upto_t', 'last_sale_date_overall'
    ]
    
    df_merged = pd.merge(
        df_total, 
        event_time_agg.drop(columns=cols_to_drop_from_time_agg), 
        on='event_name', 
        how='left'
    )
    
    return df_merged 