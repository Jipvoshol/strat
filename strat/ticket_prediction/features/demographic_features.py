import pandas as pd
import numpy as np

def engineer_demographic_features(
    tickets_upto_t: pd.DataFrame,
    df_merged: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert demografische features.
    
    Parameters:
        tickets_upto_t: Tickets tot aan tijdstip T
        df_merged: Gemergde DataFrame tot nu toe
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        DataFrame met demografische features toegevoegd
    """
    if verbose:
        print("[engineer_demographic_features] Stap 6: Demografische & Event Features...")
    
    # Start met events in df_merged
    demo_agg = pd.DataFrame({'event_name': df_merged['event_name'].unique()})
    
    # Vereiste kolommen
    required_demo = {'age', 'gender', 'city_standardized'}
    demo_cols_created = [
        'avg_age', 'std_age', 'male_percentage', 'female_percentage', 
        'nb_percentage', 'unique_cities', 'log_unique_cities', 'main_city_buyer_ratio'
    ]
    
    # Check of we de vereiste kolommen hebben
    if required_demo.issubset(tickets_upto_t.columns) and not tickets_upto_t.empty:
        # Maak kopie om waarschuwing te voorkomen
        tickets_upto_t_demo = tickets_upto_t.copy()
        tickets_upto_t_demo.loc[:, 'gender_std'] = (
            tickets_upto_t_demo['gender']
            .str.lower()
            .replace({'female': 'woman', 'male': 'man', 'nonbinary': 'nonbinary'})
            .fillna('unknown')
        )
        
        # Bereken main city ratio
        city_counts = tickets_upto_t_demo.groupby(['event_name', 'city_standardized']).size().reset_index(name='count')
        if not city_counts.empty:
            city_counts = city_counts.sort_values(['event_name', 'count'], ascending=[True, False])
            main_cities = city_counts.drop_duplicates(subset='event_name', keep='first')
            event_totals = tickets_upto_t_demo.groupby('event_name').size().reset_index(name='total_count')
            main_cities = pd.merge(main_cities, event_totals, on='event_name', how='left')
            # Voorkom delen door nul
            main_cities['main_city_buyer_ratio'] = main_cities['count'].div(main_cities['total_count']).fillna(0)
        else:
            main_cities = pd.DataFrame(columns=['event_name', 'main_city_buyer_ratio'])
        
        # Aggregeer demografie
        df_agg_demo = tickets_upto_t_demo.groupby('event_name').agg(
            avg_age=('age', 'mean'),
            std_age=('age', 'std'),
            male_percentage=('gender_std', lambda x: (x == 'man').mean() * 100),
            female_percentage=('gender_std', lambda x: (x == 'woman').mean() * 100),
            nb_percentage=('gender_std', lambda x: (x == 'nonbinary').mean() * 100),
            unique_cities=('city_standardized', 'nunique')
        ).reset_index()
        
        # Merge ratio en voeg log unique cities toe
        df_agg_demo = pd.merge(df_agg_demo, main_cities[['event_name', 'main_city_buyer_ratio']], on='event_name', how='left')
        df_agg_demo['log_unique_cities'] = np.log1p(df_agg_demo['unique_cities'])
        
        # Merge met hoofdframe
        demo_agg = pd.merge(demo_agg, df_agg_demo, on='event_name', how='left')
        
        # Vul NaNs
        for col in ['avg_age', 'std_age']:
            demo_agg[col] = demo_agg[col].fillna(demo_agg[col].median())
        
        # Vul overige met 0
        for col in ['male_percentage', 'female_percentage', 'nb_percentage', 'unique_cities', 'log_unique_cities', 'main_city_buyer_ratio']:
            if col in demo_agg.columns:
                demo_agg[col] = demo_agg[col].fillna(0)
            else:
                demo_agg[col] = 0  # Zorg dat kolom bestaat
        
        if verbose:
            print("Demografische features geaggregeerd.")
    else:
        if verbose:
            print("Onvoldoende data voor demo features.")
        # Vul alle defaults met 0
        for col in demo_cols_created:
            demo_agg[col] = 0
    
    # Merge demo features
    df_merged = pd.merge(df_merged, demo_agg, on='event_name', how='left')
    
    return df_merged 