import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from ticket_prediction.utils.text_processing import normalize_text
from ticket_prediction.data.preprocessing import find_artist_column

def engineer_artist_features(
    line_up_df: pd.DataFrame,
    artists_df: Optional[pd.DataFrame],
    df_merged: pd.DataFrame,
    forecast_days: int,
    star_artist_percentile: float = 0.80,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert artist-gebaseerde features.
    
    Parameters:
        line_up_df: Line-up DataFrame
        artists_df: Artists insights DataFrame
        df_merged: Gemergde DataFrame tot nu toe
        forecast_days: Aantal dagen voor forecast
        star_artist_percentile: Percentile voor star artist bepaling
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        DataFrame met artist features toegevoegd
    """
    if verbose:
        print("[engineer_artist_features] Stap 5: Artist Features...")
    
    # Zoek artist kolommen
    lineup_artist_col = find_artist_column(line_up_df)
    artist_col = find_artist_column(artists_df, ['artiest naam', 'artiest_naam', 'artist', 'naam', 'name'])
    
    # Definieer metrics
    metrics_to_log = {'SpotifyFollowers', 'TikTokViews', 'DeezerFans', 'ChartmetricScore', 'SpotifyPopularity', 'num_artists'}
    aggregations = ['mean', 'max', 'sum', 'std']
    possible_metrics = ['ChartmetricScore', 'SpotifyPopularity', 'SpotifyFollowers', 'TikTokViews', 'DeezerFans']
    possible_artist_cols = ['num_artists', 'total_playing_time_hours', 'avg_set_duration_minutes']
    feature_mappings = {}
    
    # Zoek kolommen in artists_df
    if artists_df is not None:
        if any('chart' in c.lower() for c in artists_df.columns):
            feature_mappings['ChartmetricScore'] = next((c for c in artists_df.columns if 'chart' in c.lower()), None)
        if any('popular' in c.lower() for c in artists_df.columns):
            feature_mappings['SpotifyPopularity'] = next((c for c in artists_df.columns if 'popular' in c.lower()), None)
        if any('follower' in c.lower() for c in artists_df.columns):
            feature_mappings['SpotifyFollowers'] = next((c for c in artists_df.columns if 'follower' in c.lower()), None)
        if any('tiktok' in c.lower() for c in artists_df.columns):
            feature_mappings['TikTokViews'] = next((c for c in artists_df.columns if 'tiktok' in c.lower()), None)
        if any('deezer' in c.lower() for c in artists_df.columns):
            feature_mappings['DeezerFans'] = next((c for c in artists_df.columns if 'deezer' in c.lower()), None)
    
    # Definieer alle mogelijke output kolommen voor default
    for metric in possible_metrics:
        if metric in feature_mappings:
            for agg in aggregations:
                possible_artist_cols.append(f'{metric}_{agg}')
            if metric in metrics_to_log:
                for agg in aggregations:
                    possible_artist_cols.append(f'log_{metric}_{agg}')
    
    possible_artist_cols.extend(['has_artist_data', 'has_star_artist'])
    default_artist_features = pd.DataFrame(0, index=range(len(df_merged)), columns=list(set(possible_artist_cols)))
    default_artist_features['event_name'] = df_merged['event_name'].unique()
    
    # Als we geen data hebben, return defaults
    if artists_df.empty or line_up_df.empty or not lineup_artist_col or not artist_col:
        if verbose:
            print("Artist/lineup data missen.")
        df_merged = pd.merge(df_merged, default_artist_features, on='event_name', how='left')
        return df_merged
    
    # Normaliseer artist namen
    line_up_df[lineup_artist_col] = line_up_df[lineup_artist_col].astype(str).apply(normalize_text)
    artists_df[artist_col] = artists_df[artist_col].astype(str).apply(normalize_text)
    
    # Selecteer relevante kolommen
    selected_cols = [artist_col] + [col for col in feature_mappings.values() if col is not None] + ['Date']
    artists_subset = artists_df[[col for col in selected_cols if col in artists_df.columns]].copy()
    
    # Filter artiestgegevens op datum
    if 'Date' in artists_subset.columns and 'event_date' in line_up_df.columns:
        # Converteer datum naar datetime formaat
        artists_subset['Date'] = pd.to_datetime(artists_subset['Date'], errors='coerce')
        line_up_df['event_date'] = pd.to_datetime(line_up_df['event_date'], errors='coerce')
        
        # Creëer filter voor de juiste artiestgegevens op basis van de predictie datum
        merged_artists_all = pd.merge(line_up_df, artists_subset, left_on=lineup_artist_col, right_on=artist_col, how='left')
        
        if not merged_artists_all.empty:
            if verbose:
                print(f"Totaal aantal artiestmetrieken vóór datumfilter: {len(merged_artists_all)}")
            
            # Bereken de datum waarop de voorspelling wordt gemaakt (T dagen voor event)
            merged_artists_all['prediction_date'] = merged_artists_all['event_date'] - pd.Timedelta(days=forecast_days)
            
            # Filter om alleen artiestgegevens te gebruiken die bekend zijn op of vóór de voorspellingsdatum
            valid_artist_data = merged_artists_all[merged_artists_all['Date'] <= merged_artists_all['prediction_date']]
            
            if valid_artist_data.empty:
                if verbose:
                    print("WAARSCHUWING: Geen artiestgegevens gevonden vóór de voorspellingsdatum. Dit kan leiden tot datalek.")
                merged_artists = merged_artists_all
            else:
                # Voor elke artiest, neem de meest recente data vóór de voorspellingsdatum
                latest_valid_metrics = valid_artist_data.sort_values('Date').groupby(['event_name', artist_col]).tail(1)
                merged_artists = latest_valid_metrics
        else:
            merged_artists = merged_artists_all
            if verbose:
                print("Geen artiestgegevens om te filteren op datum.")
    else:
        # Als datumkolommen ontbreken, voer gewone merge uit
        if verbose:
            print("WAARSCHUWING: Kan geen datumfiltering toepassen op artiestgegevens. 'Date' of 'event_date' kolom ontbreekt.")
        merged_artists = pd.merge(line_up_df, artists_subset, left_on=lineup_artist_col, right_on=artist_col, how='left')
    
    # Zoek event kolom
    event_col = next((col for col in merged_artists.columns if col.lower() in ['event_name', 'event']), None)
    
    if not event_col:
        if verbose:
            print("Geen event kolom.")
        df_merged = pd.merge(df_merged, default_artist_features, on='event_name', how='left')
        return df_merged
    
    # Hernoem naar event_name
    merged_artists.rename(columns={event_col: 'event_name'}, inplace=True)
    merged_artists['event_name'] = merged_artists['event_name'].astype(str).apply(normalize_text)
    merged_artists = merged_artists[merged_artists['event_name'].isin(df_merged['event_name'])]
    
    # Maak aggregatie functies
    agg_funcs = {'num_artists': (lineup_artist_col, 'nunique')}
    artist_cols_to_process_for_agg = {}
    
    for metric_name, source_col in feature_mappings.items():
        if source_col is not None and source_col in merged_artists.columns:
            merged_artists[source_col] = pd.to_numeric(merged_artists[source_col], errors='coerce')
            artist_cols_to_process_for_agg[metric_name] = source_col
            for agg in aggregations:
                agg_funcs[f'{metric_name}_{agg}'] = (source_col, agg)
    
    # Aggregeer features
    valid_aggregation = False
    if 'event_name' in merged_artists.columns and agg_funcs:
        try:
            combined_features = merged_artists.groupby('event_name').agg(**agg_funcs).reset_index()
            valid_aggregation = True
            if verbose:
                print("Artist features geaggregeerd.")
        except Exception as e_agg:
            if verbose:
                print(f"FOUT agg artist features: {e_agg}")
            combined_features = default_artist_features.copy()
    else:
        if verbose:
            print("Kan artist features niet aggregeren.")
        combined_features = default_artist_features.copy()
    
    if valid_aggregation:
        # Log transformaties
        if 'num_artists' in combined_features.columns:
            combined_features['log_num_artists'] = np.log1p(combined_features['num_artists'].fillna(0))
        
        for metric_name in artist_cols_to_process_for_agg.keys():
            if metric_name in metrics_to_log:
                for agg in aggregations:
                    orig_col = f'{metric_name}_{agg}'
                    log_col = f'log_{orig_col}'
                    if orig_col in combined_features.columns:
                        combined_features[log_col] = np.log1p(combined_features[orig_col].fillna(0))
                    elif log_col not in combined_features.columns:
                        combined_features[log_col] = 0
        
        # Has_artist_data & Has_star_artist
        main_metric_orig = 'SpotifyFollowers_max'
        main_metric_log = 'log_SpotifyFollowers_max'
        
        if main_metric_orig in combined_features.columns:
            combined_features['has_artist_data'] = (combined_features[main_metric_orig].fillna(0) > 0).astype(int)
        else:
            combined_features['has_artist_data'] = 0
        
        star_threshold = 0
        if main_metric_log in combined_features.columns:
            followers_data = combined_features.loc[combined_features['has_artist_data'] == 1, main_metric_log]
            if not followers_data.empty:
                star_threshold = followers_data.quantile(star_artist_percentile)
        
        combined_features['has_star_artist'] = (combined_features[main_metric_log].fillna(0) > star_threshold).astype(int)
        
        # Merge & Vul NaNs
        df_merged = pd.merge(df_merged, combined_features, on='event_name', how='left')
        all_artist_agg_cols = combined_features.columns.drop('event_name')
        for col in all_artist_agg_cols:
            if col in df_merged.columns:
                if pd.api.types.is_numeric_dtype(df_merged[col]):
                    df_merged[col] = df_merged[col].fillna(0)
    else:
        df_merged = pd.merge(df_merged, default_artist_features, on='event_name', how='left')
        if verbose:
            print("Default artist features gemerged.")
    
    return df_merged 