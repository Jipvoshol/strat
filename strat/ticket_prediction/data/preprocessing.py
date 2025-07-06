import pandas as pd
from typing import List, Optional

def find_artist_column(df: pd.DataFrame, common_names: List[str] = None) -> Optional[str]:
    """
    Zoekt naar de kolomnaam die waarschijnlijk artiestennamen bevat.
    
    Parameters:
        df (pd.DataFrame): DataFrame om te doorzoeken
        common_names (List[str], optional): Lijst met gebruikelijke namen voor artiestkolommen
        
    Returns:
        Optional[str]: Naam van de gevonden kolom, of None
    """
    if common_names is None:
        common_names = ['artist', 'artist_name', 'artiest', 'artiest_naam', 'artist naam', 'naam', 'name']
    
    # Zoek eerst exact in lowercase
    for col in df.columns:
        if col.lower() in common_names:
            return col
    
    # Anders zoek naar kolommen die een van de namen bevatten
    for col in df.columns:
        for name in common_names:
            if name in col.lower():
                return col
    
    return None

def prepare_weather_data(tickets_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Verbeterde versie: Prepareert weersgegevens voor gebruik in het model.
    
    Parameters:
        tickets_df (pd.DataFrame): DataFrame met tickets data
        verbose (bool): Of debug informatie geprint moet worden
        
    Returns:
        pd.DataFrame: Tickets DataFrame met geprepareerde weersgegevens
    """
    tickets_df = tickets_df.copy()
    
    weather_cols = ['t_max', 't_min', 'rain_fall', 'max_wind']
    has_weather_data = all(col in tickets_df.columns for col in weather_cols)
    
    if not has_weather_data:
        if verbose:
            print("Geen weer kolommen gevonden in de dataset.")
        for col in weather_cols:
            tickets_df[col] = 0
        return tickets_df
    
    for col in weather_cols:
        tickets_df[col] = pd.to_numeric(tickets_df[col], errors='coerce')
        if verbose:
            non_null = tickets_df[col].notna().sum()
            percent = 100 * non_null / len(tickets_df)
            print(f"Kolom {col}: {non_null} niet-null waarden ({percent:.1f}%)")
    
    if 'event_name' not in tickets_df.columns:
        if verbose:
            print("Kolom 'event_name' niet gevonden, kan weer niet per event aggregeren")
        for col in weather_cols:
            median_val = tickets_df[col].median()
            tickets_df[col] = tickets_df[col].fillna(median_val if not pd.isna(median_val) else 0)
        return tickets_df
    
    tickets_df['event_name'] = tickets_df['event_name'].astype(str).str.lower().str.strip()
    
    weather_by_event = {}
    for event in tickets_df['event_name'].unique():
        event_mask = tickets_df['event_name'] == event
        event_data = tickets_df.loc[event_mask, weather_cols]
        medians = event_data.median()
        weather_by_event[event] = {col: medians[col] for col in weather_cols}
    
    for event, weather in weather_by_event.items():
        event_mask = tickets_df['event_name'] == event
        for col in weather_cols:
            if pd.notna(weather[col]):
                tickets_df.loc[event_mask & tickets_df[col].isna(), col] = weather[col]
    
    for col in weather_cols:
        remaining_nulls = tickets_df[col].isna().sum()
        if remaining_nulls > 0:
            median_val = tickets_df[col].median()
            if pd.isna(median_val):
                median_val = 0
            tickets_df.loc[tickets_df[col].isna(), col] = median_val
            if verbose:
                print(f"Opgevuld: {remaining_nulls} ontbrekende waarden in {col} met {median_val}")
    
    if 't_max' in tickets_df.columns and 't_min' in tickets_df.columns:
        tickets_df['temp_contrast'] = tickets_df['t_max'] - tickets_df['t_min']
    
    if 'rain_fall' in tickets_df.columns:
        tickets_df['is_rainy'] = (tickets_df['rain_fall'] > 1.0).astype(int)
    
    if 'event_date' in tickets_df.columns:
        try:
            if not pd.api.types.is_datetime64_dtype(tickets_df['event_date']):
                tickets_df['event_date'] = pd.to_datetime(tickets_df['event_date'], errors='coerce')
            
            tickets_df['event_month'] = tickets_df['event_date'].dt.month
            tickets_df['event_season'] = tickets_df['event_date'].dt.month % 12 // 3 + 1
        except Exception as e:
            if verbose:
                print(f"Fout bij toevoegen seizoen: {e}")
    
    if verbose:
        for col in weather_cols:
            nulls = tickets_df[col].isna().sum()
            if nulls > 0:
                print(f"WAARSCHUWING: Nog steeds {nulls} NaN waarden in {col}!")
            else:
                print(f"Kolom {col}: Succesvol gevuld, geen NaN waarden meer.")
    
    return tickets_df 