import pandas as pd
import numpy as np
from predict import (
    predict_event_tickets_live, 
    find_and_load_models, 
    load_data, 
    get_real_cities_present_in_data, 
    normalize_text,
    predict_tickets,
    engineer_features
)
import pickle
import os

def deep_analysis_predictions():
    """
    Diepgaande analyse van voorspellingen vs. cumulatieve verkoop
    """
    print("=== DIEPGAANDE ANALYSE VAN VOORSPELLINGEN ===")
    
    # 1. Modellen en data laden
    print("\n1. Laden van modellen en data...")
    find_and_load_models(verbose=False)
    base_tickets_df, _, base_artists_fb_df = load_data()
    known_cities = get_real_cities_present_in_data(base_tickets_df, verbose=False)
    
    # 2. Selecteer verschillende events voor testing
    print("\n2. Selecteren van test events...")
    # Kies events met verschillende karakteristieken
    test_events = [
        "19.07.25 Palet Mini Festival - 2025-07-19",
        "Frenzy Courtyard IV - 2025-07-20", 
        "De Binnenstad - 2025-07-26",
        "Dekmantel ...IsBurning - 2025-08-01",
        "Everyday People Amsterdam - 2025-08-10"
    ]
    
    # 3. Test verschillende T-waarden
    test_T_values = [1, 5, 10, 15, 20, 30]
    
    results = []
    
    for event_name in test_events:
        print(f"\n--- Testing event: {event_name} ---")
        
        # Vind event data
        normalized_event_name = normalize_text(event_name)
        event_data_row = base_tickets_df[
            base_tickets_df['event_name'].astype(str).apply(normalize_text) == normalized_event_name
        ]
        
        if event_data_row.empty:
            print(f"Event niet gevonden: {event_name}")
            continue
            
        event_data_series = event_data_row.iloc[0]
        
        # Krijg historische data voor dit event
        event_specific_history = base_tickets_df[
            base_tickets_df['event_name'].astype(str).apply(normalize_text) == normalized_event_name
        ].copy()
        
        print(f"Totale historische tickets voor dit event: {len(event_specific_history)}")
        
        for T in test_T_values:
            try:
                # Bereken T_days_ago
                event_start_date = pd.to_datetime(event_data_series['first_event_date_start'], utc=True)
                T_days_ago = event_start_date.normalize() - pd.Timedelta(days=T)
                
                # Filter historische data tot T dagen geleden
                hist_until_T = event_specific_history[
                    event_specific_history['verkoopdatum'] <= T_days_ago
                ].copy()
                
                cum_sales_at_T = len(hist_until_T)
                
                # Doe voorspelling
                prediction, message = predict_event_tickets_live(
                    selected_event_name=event_name,
                    selected_event_data=event_data_series,
                    manual_lineup_str="test_artist",
                    base_tickets_df_context=base_tickets_df,
                    base_artists_df_for_fallback=base_artists_fb_df,
                    known_cities_context=known_cities,
                    verbose=False
                )
                
                # Check voor probleem
                is_problem = prediction < cum_sales_at_T
                
                result = {
                    'event_name': event_name,
                    'T': T,
                    'T_days_ago': T_days_ago.date(),
                    'cumulative_sales_at_T': cum_sales_at_T,
                    'prediction': prediction,
                    'is_problem': is_problem,
                    'difference': prediction - cum_sales_at_T,
                    'message': message
                }
                
                results.append(result)
                
                if is_problem:
                    print(f"  🚨 PROBLEEM bij T={T}: voorspelling={prediction} < cum_sales={cum_sales_at_T}")
                else:
                    print(f"  ✅ OK bij T={T}: voorspelling={prediction} >= cum_sales={cum_sales_at_T}")
                    
            except Exception as e:
                print(f"  ❌ Error bij T={T}: {e}")
    
    # 4. Analyseer resultaten
    print("\n=== ANALYSE VAN RESULTATEN ===")
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        total_tests = len(df_results)
        problem_tests = df_results['is_problem'].sum()
        
        print(f"Totaal aantal tests: {total_tests}")
        print(f"Tests met problemen: {problem_tests} ({problem_tests/total_tests*100:.1f}%)")
        
        if problem_tests > 0:
            print("\nPROBLEEM CASES:")
            problem_cases = df_results[df_results['is_problem']]
            for _, row in problem_cases.iterrows():
                print(f"  Event: {row['event_name']}")
                print(f"    T={row['T']}, Cum={row['cumulative_sales_at_T']}, Pred={row['prediction']}, Diff={row['difference']}")
        
        # Groepeer per T-waarde
        print("\nANALYSE PER T-WAARDE:")
        for T in sorted(df_results['T'].unique()):
            T_data = df_results[df_results['T'] == T]
            problems = T_data['is_problem'].sum()
            total = len(T_data)
            print(f"  T={T}: {problems}/{total} problemen ({problems/total*100:.1f}%)")
        
        return df_results
    else:
        print("Geen resultaten om te analyseren")
        return pd.DataFrame()

def analyze_model_internals():
    """
    Analyseer wat de modellen daadwerkelijk voorspellen
    """
    print("\n=== ANALYSE VAN MODEL INTERNALS ===")
    
    # Laad een model en inspecteer het
    from predict import best_models
    
    if not best_models:
        print("Geen modellen geladen!")
        return
    
    # Kies T=10 model voor analyse
    T_to_analyze = 10
    if T_to_analyze not in best_models:
        T_to_analyze = list(best_models.keys())[0]
    
    model_pipeline, target_is_transformed = best_models[T_to_analyze]
    
    print(f"Analyseren van model T={T_to_analyze}")
    print(f"Target is getransformeerd: {target_is_transformed}")
    print(f"Model pipeline steps: {[step[0] for step in model_pipeline.steps]}")
    
    # Bekijk de preprocessor features
    if hasattr(model_pipeline, 'named_steps') and 'preprocessor' in model_pipeline.named_steps:
        preprocessor = model_pipeline.named_steps['preprocessor']
        if hasattr(preprocessor, 'feature_names_in_'):
            features = list(preprocessor.feature_names_in_)
            print(f"Model input features ({len(features)}):")
            for i, feature in enumerate(features):
                print(f"  {i+1}: {feature}")
    
    return model_pipeline, target_is_transformed

def inspect_training_data_patterns():
    """
    Inspecteer patronen in de training data
    """
    print("\n=== INSPECTIE VAN TRAINING DATA PATRONEN ===")
    
    # Laad tickets data
    tickets = pd.read_csv('tickets_processed.csv')
    print(f"Totaal aantal tickets in training data: {len(tickets)}")
    
    # Analyseer per event: wat is de verhouding tussen vroege verkoop en totale verkoop?
    print("\nAnalyse vroege vs. totale verkoop per event...")
    
    events_analysis = []
    
    # Selecteer events met substantiële verkoop
    event_counts = tickets['event_name'].value_counts()
    large_events = event_counts[event_counts >= 50].index[:10]  # Top 10 grote events
    
    for event_name in large_events:
        event_tickets = tickets[tickets['event_name'] == event_name].copy()
        
        if len(event_tickets) == 0:
            continue
            
        # Converteer datums
        event_tickets['verkoopdatum'] = pd.to_datetime(event_tickets['verkoopdatum'], errors='coerce')
        event_tickets['first_event_date_start'] = pd.to_datetime(event_tickets['first_event_date_start'], errors='coerce')
        
        # Filter geldige datums
        valid_tickets = event_tickets.dropna(subset=['verkoopdatum', 'first_event_date_start'])
        
        if len(valid_tickets) == 0:
            continue
        
        event_start = valid_tickets['first_event_date_start'].iloc[0]
        total_tickets = len(valid_tickets)
        
        # Bereken verkoop op verschillende T-punten
        analysis_result = {
            'event_name': event_name,
            'total_tickets': total_tickets,
            'event_date': event_start.date() if pd.notna(event_start) else None
        }
        
        for T in [1, 5, 10, 20, 30]:
            T_days_ago = event_start - pd.Timedelta(days=T)
            tickets_until_T = valid_tickets[valid_tickets['verkoopdatum'] <= T_days_ago]
            
            analysis_result[f'tickets_at_T{T}'] = len(tickets_until_T)
            analysis_result[f'percentage_at_T{T}'] = len(tickets_until_T) / total_tickets * 100
        
        events_analysis.append(analysis_result)
    
    df_analysis = pd.DataFrame(events_analysis)
    
    if len(df_analysis) > 0:
        print("\nVerkoop patronen in training data:")
        print("Event | Total | T30 | T20 | T10 | T5 | T1")
        print("-" * 60)
        
        for _, row in df_analysis.iterrows():
            event_short = row['event_name'][:30] + "..." if len(row['event_name']) > 30 else row['event_name']
            print(f"{event_short:<33} | {row['total_tickets']:4d} | "
                  f"{row['percentage_at_T30']:5.1f}% | {row['percentage_at_T20']:5.1f}% | "
                  f"{row['percentage_at_T10']:5.1f}% | {row['percentage_at_T5']:5.1f}% | {row['percentage_at_T1']:5.1f}%")
    
    return df_analysis

def test_extreme_scenarios():
    """
    Test extreme scenarios die problemen zouden kunnen veroorzaken
    """
    print("\n=== TEST VAN EXTREME SCENARIOS ===")
    
    from predict import best_models, predict_tickets, engineer_features
    import numpy as np
    
    # Laad basis data
    base_tickets_df, _, base_artists_fb_df = load_data()
    known_cities = get_real_cities_present_in_data(base_tickets_df, verbose=False)
    
    # Test 1: Event met ZEER hoge cumulatieve verkoop
    print("\n1. Test met zeer hoge cumulatieve verkoop...")
    large_event = "Dekmantel IsBurning - 2023-08-05"  # Een groot event uit de training data
    
    large_event_data = base_tickets_df[
        base_tickets_df['event_name'].astype(str).apply(normalize_text) == normalize_text(large_event)
    ]
    
    if not large_event_data.empty:
        # Simuleer scenario met alle tickets al verkocht op T=10
        print(f"Event gevonden: {large_event} met {len(large_event_data)} tickets")
        
        # Test het model direct met hoge cumulative_sales_at_t
        test_features = pd.DataFrame({
            'log_cumulative_sales_at_t': [np.log1p(5000)],  # Zeer hoge waarde
            'log_sales_last_3_days': [0.0],
            'log_avg_daily_sales_before_t': [np.log1p(100)],
            'avg_acceleration': [0.0],
            'SpotifyPopularity_max': [50],
            'avg_age': [30],
            't_max': [20],
            'month_cos': [0.5],
            'day_of_week_cos': [0.5]
        })
        
        # Test model T=10
        if 10 in best_models:
            model, is_transformed = best_models[10]
            raw_pred = model.predict(test_features)[0]
            final_pred = np.expm1(raw_pred) if is_transformed else raw_pred
            
            print(f"  Input: cumulative_sales = 5000")
            print(f"  Raw prediction: {raw_pred:.4f}")
            print(f"  Final prediction: {final_pred:.1f}")
            print(f"  Is problematic: {final_pred < 5000}")
    
    # Test 2: Directe model test met verschillende log_cumulative_sales_at_t waarden
    print("\n2. Test model response tot verschillende cumulatieve verkoop...")
    
    if 10 in best_models:
        model, is_transformed = best_models[10]
        
        test_cumulative_values = [0, 10, 50, 100, 500, 1000, 2000, 5000]
        
        for cum_sales in test_cumulative_values:
            test_features = pd.DataFrame({
                'log_cumulative_sales_at_t': [np.log1p(cum_sales)],
                'log_sales_last_3_days': [0.0],
                'log_avg_daily_sales_before_t': [np.log1p(max(1, cum_sales/10))],
                'avg_acceleration': [0.0],
                'SpotifyPopularity_max': [50],
                'avg_age': [30],
                't_max': [20],
                'month_cos': [0.5],
                'day_of_week_cos': [0.5]
            })
            
            raw_pred = model.predict(test_features)[0]
            final_pred = np.expm1(raw_pred) if is_transformed else raw_pred
            is_problem = final_pred < cum_sales
            
            status = "🚨 PROBLEEM" if is_problem else "✅ OK"
            print(f"  Cum={cum_sales:4d} → Pred={final_pred:6.1f} | {status}")
    
    # Test 3: Inspecteer een van de training voorbeelden
    print("\n3. Inspecteer training voorbeelden...")
    
    # Neem een willekeurig event en kijk hoe het model daarop reageert
    sample_event = "Lofi King's Day - 2023-04-27"
    sample_data = base_tickets_df[
        base_tickets_df['event_name'].astype(str).apply(normalize_text) == normalize_text(sample_event)
    ]
    
    if not sample_data.empty:
        print(f"Sample event: {sample_event} ({len(sample_data)} tickets)")
        
        # Simuleer verschillende T-momenten
        event_start = pd.to_datetime(sample_data['first_event_date_start'].iloc[0])
        
        for T in [30, 20, 10, 5, 1]:
            T_date = event_start - pd.Timedelta(days=T)
            tickets_at_T = sample_data[sample_data['verkoopdatum'] <= T_date]
            cum_sales_at_T = len(tickets_at_T)
            
            if cum_sales_at_T > 0:
                # Test model prediction
                test_features = pd.DataFrame({
                    'log_cumulative_sales_at_t': [np.log1p(cum_sales_at_T)],
                    'log_sales_last_3_days': [0.0],
                    'log_avg_daily_sales_before_t': [np.log1p(max(1, cum_sales_at_T/T))],
                    'avg_acceleration': [0.0],
                    'SpotifyPopularity_max': [50],
                    'avg_age': [28],
                    't_max': [15],
                    'month_cos': [0.0],  # April
                    'day_of_week_cos': [0.0]  # Approximately
                })
                
                if 10 in best_models:  # Use T=10 model for consistency
                    model, is_transformed = best_models[10]
                    raw_pred = model.predict(test_features)[0]
                    final_pred = np.expm1(raw_pred) if is_transformed else raw_pred
                    is_problem = final_pred < cum_sales_at_T
                    
                    status = "🚨" if is_problem else "✅"
                    print(f"  T={T:2d}: Cum={cum_sales_at_T:4d} → Pred={final_pred:6.1f} {status}")

if __name__ == "__main__":
    # Voer alle analyses uit
    results_df = deep_analysis_predictions()
    model_pipeline, target_transformed = analyze_model_internals()
    training_patterns_df = inspect_training_data_patterns()
    test_extreme_scenarios()
    
    print("\n=== SAMENVATTING ===")
    print("Alle analyses voltooid. Bekijk de output hierboven voor details.")