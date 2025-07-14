#!/usr/bin/env python3
"""
DATASET-SPECIFIEKE PREPROCESSING VERBETERING
Voorzichtige cleaning die rekening houdt met verschillen per event_source
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_data_per_source(df):
    """Analyseer data problemen per event source"""
    print("=== DATA ANALYSE PER EVENT SOURCE ===")
    
    for source in df['event_source'].unique():
        subset = df[df['event_source'] == source]
        print(f"\n{source} ({len(subset):,} rows):")
        
        # Product value issues
        neg_values = (subset['product_value'] < 0).sum()
        zero_values = (subset['product_value'] == 0).sum()
        print(f"  Product_value: {neg_values:,} negative, {zero_values:,} zero")
        
        # Missing values
        age_missing = subset['age'].isna().sum()
        gender_missing = subset['gender'].isna().sum()
        print(f"  Missing: age={age_missing:,} ({age_missing/len(subset)*100:.1f}%), gender={gender_missing:,} ({gender_missing/len(subset)*100:.1f}%)")
        
        # Date issues
        verkoop_dt = pd.to_datetime(subset['verkoopdatum'], errors='coerce')
        event_dt = pd.to_datetime(subset['event_date'], errors='coerce')
        verkoop_success = verkoop_dt.notna().sum()
        event_success = event_dt.notna().sum()
        print(f"  Date success: verkoop={verkoop_success:,}/{len(subset):,} ({verkoop_success/len(subset)*100:.1f}%), event={event_success:,}/{len(subset):,} ({event_success/len(subset)*100:.1f}%)")
    
    return df

def clean_dataset_carefully(df):
    """Voorzichtige cleaning per event source"""
    print("\n=== VOORZICHTIGE CLEANING PER EVENT SOURCE ===")
    
    df_clean = df.copy()
    total_removed = 0
    
    # 1. PRODUCT VALUE CLEANING - alleen echte problemen
    print("\n1. Product Value Cleaning:")
    for source in df_clean['event_source'].unique():
        mask = df_clean['event_source'] == source
        subset = df_clean[mask]
        
        # Alleen negatieve waarden verwijderen (zero is OK voor sommige events)
        negative_mask = (subset['product_value'] < 0)
        removed = negative_mask.sum()
        
        if removed > 0:
            df_clean = df_clean[~(mask & negative_mask)]
            total_removed += removed
            print(f"  {source}: {removed:,} negatieve product_value verwijderd")
    
    # 2. GENDER STANDARDIZATION - alleen als er data is
    print("\n2. Gender Standardization:")
    for source in df_clean['event_source'].unique():
        mask = df_clean['event_source'] == source
        subset = df_clean[mask]
        
        gender_before = subset['gender'].value_counts(dropna=False)
        if len(gender_before) > 0:
            print(f"  {source} gender before: {dict(gender_before)}")
            
            # Standaardiseer gender values
            gender_mapping = {
                'male': 'man',
                'female': 'woman', 
                'woman': 'woman',
                'man': 'man',
                'nonbinary': 'nonbinary',
                'non binary': 'nonbinary'
            }
            
            df_clean.loc[mask, 'gender'] = df_clean.loc[mask, 'gender'].map(gender_mapping)
            
            gender_after = df_clean.loc[mask, 'gender'].value_counts(dropna=False)
            print(f"  {source} gender after: {dict(gender_after)}")
    
    # 3. DATE CLEANING - voorzichtig per source
    print("\n3. Date Cleaning:")
    for source in df_clean['event_source'].unique():
        mask = df_clean['event_source'] == source
        subset = df_clean[mask]
        
        # Converteer datums
        verkoop_dt = pd.to_datetime(subset['verkoopdatum'], errors='coerce')
        event_dt = pd.to_datetime(subset['event_date'], errors='coerce')
        
        # Verwijder alleen rijen met beide datums invalid
        invalid_dates = verkoop_dt.isna() & event_dt.isna()
        removed = invalid_dates.sum()
        
        if removed > 0:
            df_clean = df_clean[~(mask & invalid_dates)]
            total_removed += removed
            print(f"  {source}: {removed:,} rijen met beide datums invalid verwijderd")
        
        # Update datums in main dataframe
        df_clean.loc[mask, 'verkoopdatum'] = verkoop_dt
        df_clean.loc[mask, 'event_date'] = event_dt
        
        # Check future sales (maar niet verwijderen - kan legitiem zijn)
        valid_dates = verkoop_dt.notna() & event_dt.notna()
        if valid_dates.sum() > 0:
            days_diff = (event_dt - verkoop_dt).dt.days
            future_sales = (days_diff < 0).sum()
            if future_sales > 0:
                print(f"  {source}: {future_sales:,} future sales detected (NIET verwijderd)")
    
    # 4. BASIC FEATURE ENGINEERING
    print("\n4. Basic Feature Engineering:")
    
    # Age categories
    df_clean['age_category'] = pd.cut(df_clean['age'], 
                                     bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Days until event (safe datetime conversion)
    verkoop_dt = pd.to_datetime(df_clean['verkoopdatum'], errors='coerce')
    event_dt = pd.to_datetime(df_clean['event_date'], errors='coerce')
    df_clean['days_until_event'] = (event_dt - verkoop_dt).dt.days
    
    # Venue type based on capacity
    df_clean['venue_type'] = df_clean['max_capacity'].apply(
        lambda x: 'club' if pd.isna(x) or x < 3000 else 'festival'
    )
    
    # Event month (safe)
    df_clean['event_month'] = event_dt.dt.month
    
    # Sale month (safe)
    df_clean['sale_month'] = verkoop_dt.dt.month
    
    print(f"\nTotaal verwijderd: {total_removed:,} rijen")
    print(f"Finale dataset: {len(df_clean):,} rijen")
    
    return df_clean

def main():
    """Hoofdfunctie voor voorzichtige preprocessing"""
    print("=== DATASET-SPECIFIEKE PREPROCESSING ===")
    
    # Laad data
    print("Laden van unified dataset...")
    df = pd.read_csv('tickets_unified_raw_fixed.csv', low_memory=False)
    print(f"Geladen: {df.shape}")
    
    # Analyseer per source
    analyze_data_per_source(df)
    
    # Voorzichtige cleaning
    df_clean = clean_dataset_carefully(df)
    
    # Finale analyse
    print("\n=== FINALE ANALYSE ===")
    print(f"Event sources na cleaning:")
    print(df_clean['event_source'].value_counts())
    
    print(f"\nEvents per source na cleaning:")
    print(df_clean.groupby('event_source')['event_name'].nunique())
    
    # Check of we festival data hebben behouden
    festival_events = df_clean[df_clean['venue_type'] == 'festival']['event_name'].nunique()
    club_events = df_clean[df_clean['venue_type'] == 'club']['event_name'].nunique()
    print(f"\nVenue types na cleaning:")
    print(f"  Festivals: {festival_events} events")
    print(f"  Clubs: {club_events} events")
    
    if festival_events == 0:
        print("🚨 WAARSCHUWING: Alle festival events zijn verdwenen!")
        return
    
    # Opslaan
    output_file = 'tickets_unified_cleaned_careful.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\n✅ Opgeslagen als: {output_file}")
    
    # Basis statistieken
    print(f"\nBasis statistieken:")
    print(f"  Totaal tickets: {len(df_clean):,}")
    print(f"  Unieke events: {df_clean['event_name'].nunique()}")
    print(f"  Gem. product_value: €{df_clean['product_value'].mean():.2f}")
    print(f"  Missing age: {df_clean['age'].isna().sum():,} ({df_clean['age'].isna().mean()*100:.1f}%)")
    print(f"  Missing gender: {df_clean['gender'].isna().sum():,} ({df_clean['gender'].isna().mean()*100:.1f}%)")

if __name__ == "__main__":
    main() 