import pandas as pd
import numpy as np

def engineer_weather_features(
    tickets: pd.DataFrame,
    df_merged: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert weer features en capacity features.
    
    Parameters:
        tickets: Hoofd tickets DataFrame
        df_merged: Gemergde DataFrame tot nu toe
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        DataFrame met weer features toegevoegd
    """
    if verbose:
        print("[engineer_weather_features] Stap 8 & 9: Max Capacity & Weer Features...")
    
    # --- Stap 8: Max Capacity Feature ---
    if 'max_capacity' in tickets.columns:
        # Gebruik .first() omdat capaciteit per event hopelijk constant is
        df_cap = tickets.groupby('event_name')['max_capacity'].first().reset_index()
        df_merged = pd.merge(df_merged, df_cap, on='event_name', how='left')
        median_cap = df_merged['max_capacity'].median()
        df_merged['max_capacity'] = df_merged['max_capacity'].fillna(
            median_cap if pd.notna(median_cap) else 1000
        )
        df_merged['log_max_capacity'] = np.log1p(df_merged['max_capacity'])
    else:
        if verbose:
            print("Kolom 'max_capacity' mist.")
        df_merged['max_capacity'] = 1700
        df_merged['log_max_capacity'] = np.log1p(1000)
    
    # --- Stap 9: Weer Features (Aggregatie op day0) ---
    weather_cols_agg = ['t_max', 't_min', 'rain_fall', 'max_wind']
    
    # Filter tickets voor dag 0 (dag van het event)
    day0 = tickets[
        (tickets['days_until_event'] <= 1) & 
        tickets['event_name'].isin(df_merged['event_name'].unique())
    ].copy()
    
    # Ensure event_name is unique before aggregation
    if not day0.empty and 'event_name' in day0.columns:
        # Take the most recent record for each event to avoid duplicates
        if 'verkoopdatum' in day0.columns:
            day0 = day0.sort_values('verkoopdatum').drop_duplicates('event_name', keep='last')
    
    # Default waarden voor weer
    weather_defaults = {
        't_max': 15, 
        't_min': 5, 
        'rain_fall': 0, 
        'max_wind': 20, 
        'event_type': 'other'
    }
    
    # We hebben maand/dag al, alleen event_type nog nodig uit day0 als die bestaat
    required_cols_weather = weather_cols_agg.copy()
    if 'event_type' in tickets.columns:
        required_cols_weather.append('event_type')
    
    if not day0.empty and all(c in day0.columns for c in required_cols_weather):
        try:
            # Select only relevant columns + event_name for merge
            cols_to_merge = ['event_name'] + required_cols_weather
            df_weather = day0[cols_to_merge].copy()
            
            # Double-check that event_name is unique
            if df_weather['event_name'].duplicated().any():
                if verbose:
                    print("WARNING: Event_name still not unique in df_weather")
                df_weather = df_weather.drop_duplicates('event_name', keep='last')
            
            # Merge with df_merged
            df_merged = pd.merge(df_merged, df_weather, on='event_name', how='left')
            if verbose:
                print("Weather features successfully merged.")
        except Exception as e_weather_agg:
            if verbose:
                print(f"ERROR weather aggregation: {e_weather_agg}")
            # Fill defaults only if column doesn't exist in df_merged
            for col, default_val in weather_defaults.items():
                if col not in df_merged.columns:
                    df_merged[col] = default_val
    else:  # No data in day0 or columns are missing
        if verbose:
            print("Insufficient data for weather aggregation. Filling with defaults.")
        for col, default_val in weather_defaults.items():
            if col not in df_merged.columns:
                df_merged[col] = default_val
    
    return df_merged 