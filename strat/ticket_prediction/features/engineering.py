import pandas as pd
import numpy as np
import traceback
from typing import Optional, List

# Import utility functions
from ticket_prediction.utils.text_processing import normalize_text
from ticket_prediction.utils.geo_utils import standardize_city_names
from ticket_prediction.data.preprocessing import prepare_weather_data, find_artist_column

def debug_print(message: str, verbose: bool = True) -> None:
    """Helper om berichten te printen als verbose True is."""
    if verbose:
        print(message)

def engineer_features(
    tickets: pd.DataFrame,
    line_up_df: pd.DataFrame,
    artists_df: Optional[pd.DataFrame],
    forecast_days: int = 7,
    known_cities: Optional[List[str]] = None,
    max_lag: int = 3,
    lag_days_sales: int = 3, 
    star_artist_percentile: float = 0.80, 
    is_prediction: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert een UITGEBREIDE set features voor latere analyse en selectie.
    - Inclusief meer artist aggregaties (mean, max, sum, std).
    - Log transformaties worden toegevoegd, originelen blijven behouden.
    - Voegt max_capacity, product_value aggregaties toe.
    - Voegt meer tijd features toe.
    - Voegt lag feature toe voor recente sales.
    - Voegt 'has_star_artist' feature toe.
    - Originele functie/variabelenamen behouden.
    
    """

    debug_print(f"\n--- Start Comprehensive Feature Engineering (+Lag/Star): T={forecast_days} ---", verbose)
    function_name = "engineer_features" 

    # --- Stap 1: Basis Voorbewerking ---
    debug_print(f"[{function_name}] Stap 1: Basis Voorbewerking...", verbose)
    tickets = tickets.copy()
    line_up_df = line_up_df.copy() if line_up_df is not None else pd.DataFrame()
    artists_df = artists_df.copy() if artists_df is not None else pd.DataFrame()

    # Filter events voor training mode: alleen events die al hebben plaatsgevonden
    if not is_prediction and 'first_event_date_start' in tickets.columns:
        try:
            # Controleer of er rijen zijn met dit veld
            if tickets['first_event_date_start'].notna().any():
                # Zorg ervoor dat de kolom een datetime is
                if not pd.api.types.is_datetime64_any_dtype(tickets['first_event_date_start']):
                    tickets['first_event_date_start'] = pd.to_datetime(tickets['first_event_date_start'], errors='coerce')
                
                # Analyse huidige datumformaten voor debugging
                if verbose:
                    sample_dates = tickets['first_event_date_start'].dropna().sample(min(3, tickets['first_event_date_start'].notna().sum()))
                    debug_print(f"Voorbeelden van datums vóór normalisatie: {sample_dates.tolist()}", verbose)
                
                # Filter op huidige datum, maar vergelijk alleen op dagniveau (niet uur/minuut)
                today = pd.Timestamp.now().normalize()  # Normaliseren om tijd-component te verwijderen
                before_filtering = tickets['event_name'].nunique()
                
                # Normaliseer event data ook naar alleen datum
                # We gebruiken dt.normalize() om alle tijdscomponenten te verwijderen
                event_dates_normalized = tickets['first_event_date_start'].dt.normalize()
                
                # Debug logging voor normalisatieproces
                if verbose:
                    norm_samples = list(zip(
                        tickets['first_event_date_start'].dropna().sample(min(3, tickets['first_event_date_start'].notna().sum())),
                        event_dates_normalized.dropna().sample(min(3, event_dates_normalized.notna().sum()))
                    ))
                    debug_print(f"Voorbeelden van normalisatie: {norm_samples}", verbose)
                
                # Filter op basis van genormaliseerde datums
                future_mask = event_dates_normalized > today
                tickets = tickets[~future_mask]
                
                # Tel hoeveel unieke events zijn verwijderd
                after_filtering = tickets['event_name'].nunique()
                
                if verbose and before_filtering > after_filtering:
                    removed = before_filtering - after_filtering
                    total_tickets_removed = future_mask.sum()
                    debug_print(f"Gefilterd: {removed} events ({total_tickets_removed} tickets) die nog niet hebben plaatsgevonden", verbose)
                    debug_print(f"Events voor filtering: {before_filtering}, na filtering: {after_filtering}", verbose)
                    
                    # Toon voorbeelden van gefilterde events voor debugging
                    if future_mask.sum() > 0:
                        future_events = tickets.loc[future_mask, 'event_name'].unique()[:3]
                        debug_print(f"Voorbeelden van gefilterde toekomstige events: {future_events}", verbose)
        except Exception as e:
            debug_print(f"Fout bij het filteren van events die nog niet hebben plaatsgevonden: {e}", verbose)
            traceback.print_exc()  # Print volledige stack trace voor debugging

    # Controleer of vereiste kolommen bestaan voordat prepare_weather wordt aangeroepen
    weather_cols_needed = ['t_max', 't_min', 'rain_fall', 'max_wind', 'event_name']
    if all(col in tickets.columns for col in weather_cols_needed):
        tickets = prepare_weather_data(tickets, verbose=False)
    else:
        debug_print("Niet alle vereiste weer/event_name kolommen aanwezig voor prepare_weather_data.", verbose)
        # Voeg ontbrekende weer kolommen toe met 0 indien nodig
        for col in ['t_max', 't_min', 'rain_fall', 'max_wind']:
            if col not in tickets.columns: tickets[col] = 0

    if 'event_name' not in tickets.columns:
        debug_print("FATAL: 'event_name' kolom essentieel en niet gevonden.", verbose)
        return pd.DataFrame() 
    tickets['event_name'] = tickets['event_name'].astype(str).apply(normalize_text)

    date_columns = ['verkoopdatum', 'first_event_date_start', 'last_event_date_end', 'event_date']
    for col in date_columns:
        if col in tickets.columns: tickets[col] = pd.to_datetime(tickets[col], errors='coerce')

    if known_cities is not None and 'city' in tickets.columns:
        tickets = standardize_city_names(tickets, known_cities, verbose=False)
    else: tickets['city_standardized'] = 'other'

    if {'first_event_date_start', 'verkoopdatum'}.issubset(tickets.columns) and pd.api.types.is_datetime64_any_dtype(tickets['first_event_date_start']) and pd.api.types.is_datetime64_any_dtype(tickets['verkoopdatum']):
        tickets['days_until_event'] = (tickets['first_event_date_start'] - tickets['verkoopdatum']).dt.days
        valid_days_mask = tickets['days_until_event'].notna()
        if not is_prediction: tickets = tickets[~(valid_days_mask & (tickets['days_until_event'] < 0))]
    else: tickets['days_until_event'] = forecast_days; valid_days_mask = pd.Series(True, index=tickets.index)
    tickets['days_until_event'].fillna(forecast_days, inplace=True) # Vul eventuele NaNs

    # --- Stap 2: Target & Basis df_total ---
    debug_print(f"[{function_name}] Stap 2: Target & Basis Aggregatie...", verbose)
    if 'tickets_sold' in tickets.columns and tickets['tickets_sold'].notna().any(): df_total = tickets.groupby('event_name')['tickets_sold'].sum().reset_index(name='full_event_tickets')
    else: df_total = tickets.groupby('event_name').size().reset_index(name='full_event_tickets')
    if not is_prediction:
  
        min_tickets_threshold = 400; before = df_total.shape[0]; df_total = df_total[df_total['full_event_tickets'] > min_tickets_threshold]; after = df_total.shape[0]
        max_tickets_threshold = 3500; df_total = df_total[df_total['full_event_tickets'] < max_tickets_threshold]; after = df_total.shape[0]
        if before > after: debug_print(f"Filtered {before - after} events met tickets < {min_tickets_threshold} of > {max_tickets_threshold}", verbose)
    if df_total.empty: debug_print("Geen events over.", verbose); return pd.DataFrame()
    tickets = tickets[tickets['event_name'].isin(df_total['event_name'])].copy() # Filter originele tickets

    # --- Stap 2.5: Bereken Lag Feature ---
    debug_print(f"[{function_name}] Stap 2.5: Lag Feature...", verbose)
    lag_sales_col = f'sales_last_{lag_days_sales}_days'; log_lag_sales_col = f'log_{lag_sales_col}' # 
    lag_mask = tickets['event_name'].isin(df_total['event_name']) & tickets['days_until_event'].notna() & (tickets['days_until_event'] >= forecast_days) & (tickets['days_until_event'] < forecast_days + lag_days_sales)
    df_lag = tickets[lag_mask].groupby('event_name').size().reset_index(name=lag_sales_col)
    df_total = pd.merge(df_total, df_lag, on='event_name', how='left')
    df_total[lag_sales_col] = df_total[lag_sales_col].fillna(0)
    df_total[log_lag_sales_col] = np.log1p(df_total[lag_sales_col])
    df_agg_base = df_total.copy()

    # Definieer subset tickets voor andere aggregaties
    tickets_upto_t = tickets[tickets['days_until_event'] >= forecast_days].copy()

    # Import feature engineering modules
    from ticket_prediction.features.time_features import engineer_time_features
    from ticket_prediction.features.sales_features import engineer_sales_features
    from ticket_prediction.features.artist_features import engineer_artist_features
    from ticket_prediction.features.demographic_features import engineer_demographic_features
    from ticket_prediction.features.ticket_features import engineer_ticket_features
    from ticket_prediction.features.weather_features import engineer_weather_features

    # --- Stap 3: Tijd-gebaseerde Features ---
    df_merged = engineer_time_features(tickets, tickets_upto_t, df_total, verbose)

    # --- Stap 4: Sales Features ---
    df_merged = engineer_sales_features(tickets_upto_t, df_merged, lag_days_sales, verbose)

    # --- Stap 5: Artist Features ---
    df_merged['forecast_days'] = forecast_days
    df_merged = engineer_artist_features(line_up_df, artists_df, df_merged, forecast_days, star_artist_percentile, verbose)

    # --- Stap 6: Demografische Features ---
    df_merged = engineer_demographic_features(tickets_upto_t, df_merged, verbose)

    # --- Stap 7: Ticket Info Features ---
    df_merged = engineer_ticket_features(tickets_upto_t, df_merged, verbose)

    # --- Stap 8 & 9: Max Capacity & Weer Features ---
    df_merged = engineer_weather_features(tickets, df_merged, verbose)

    # --- Stap 10: Hernoem en Finale Check ---
    df_merged['forecast_days'] = forecast_days
    cols_to_drop_final = ['gender_std']
    for col in cols_to_drop_final:
        if col in df_merged.columns:
            df_merged.drop(columns=[col], inplace=True, errors='ignore')

    final_cols = df_merged.columns.drop('full_event_tickets', errors='ignore').tolist()
    debug_print(f"Feature engineering (comprehensive+lag) voltooid. Shape: {df_merged.shape}", verbose)
    const_cols = [col for col in final_cols if df_merged[col].nunique(dropna=False) <= 1 and col != 'forecast_days']
    if const_cols and verbose:
        debug_print(f"WAARSCHUWING: Constante kolommen: {const_cols}", verbose)

    return df_merged 