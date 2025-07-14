import pandas as pd
import sys
import os
from typing import Dict, Tuple, List, Optional
from sklearn.pipeline import Pipeline

# Add project root to path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all modules
from ticket_prediction.data.loader import load_data
from ticket_prediction.data.preprocessing import find_artist_column
from ticket_prediction.utils.text_processing import normalize_text
from ticket_prediction.utils.geo_utils import get_real_cities_present_in_data
from ticket_prediction.utils.model_persistence import save_model, load_model
from ticket_prediction.features.engineering import engineer_features
from ticket_prediction.models.training import run_model_comparison
from ticket_prediction.models.prediction import predict_tickets, get_future_events
from ticket_prediction.config.constants import best_models

def main(
    tickets_path: str,
    line_up_path: str,
    artists_path: str,
    T_values: List[int],
    fuzzy_match_cutoff: int = 90,
    verbose: bool = True
) -> Dict[int, Tuple[Pipeline, bool]]:
    """
    Hoofd functie voor het laden van data, trainenen van modellen en maken van voorspellingen.
    Retourneert dictionary met getrainde modellen en transformatie-status per T.
    """
    # Laad datasets
    try:
        tickets, line_up, artists = load_data(tickets_path, line_up_path, artists_path, verbose)
        
        # Direct normalisatie van artiestkolommen na inladen
        if 'artist' in line_up.columns:
            line_up['artist'] = line_up['artist'].astype(str).apply(normalize_text)

        if 'Artist' in artists.columns:
            # Hernoem naar 'artist' voor consistentie
            artists.rename(columns={'Artist': 'artist'}, inplace=True)
            artists['artist'] = artists['artist'].astype(str).apply(normalize_text)

        # Controleer overlap tussen datasets na normalisatie
        if 'artist' in line_up.columns and 'artist' in artists.columns:
            lineup_artists = set(line_up['artist'].unique())
            insight_artists = set(artists['artist'].unique())
            overlap = lineup_artists.intersection(insight_artists)
            if len(lineup_artists) > 0:
                if verbose:
                    print(f"\n=== Initiële Overlap van Artiesten ===")
                    print(f"Overlap van artiesten tussen datasets: {len(overlap)} van {len(lineup_artists)} ({len(overlap)/len(lineup_artists)*100:.1f}%)")
            else:
                if verbose:
                    print("Geen artiesten in line-up data om overlap te checken.")
    except Exception as e:
        print(f"Fout bij het laden van de CSV bestanden: {e}")
        return {}

    # Voorverwerking van dataframes
    if verbose:
        print("\n--- Voorverwerking van dataframes ---")

    # Verkrijg geldige steden
    known_cities = get_real_cities_present_in_data(tickets, verbose=verbose)

    # Train modellen voor elke forecast horizon
    for T in T_values:
        if verbose:
            print(f"\n=== Model Training voor T={T} dagen ===")

        # Feature engineering
        df_features = engineer_features(
            tickets=tickets.copy(),
            line_up_df=line_up.copy(),
            artists_df=artists.copy(),
            forecast_days=T,
            known_cities=known_cities,
            max_lag=3,
            is_prediction=False,
            verbose=verbose
        )

        # Verwijder rijen met ontbrekende target waarden
        if 'full_event_tickets' in df_features.columns:
            before = df_features.shape[0]
            df_features = df_features.dropna(subset=['full_event_tickets'])
            after = df_features.shape[0]
            if before > after:
                if verbose:
                    print(f"Verwijderd {before - after} rijen met missende target waarden.")
            
            # Controleer overlap van artiesten na ticket filtering
            if verbose and 'event_name' in line_up.columns:
                event_names_after_filtering = set(df_features['event_name'].unique())
                filtered_lineup = line_up[line_up['event_name'].isin(event_names_after_filtering)]
                filtered_lineup_artists = set(filtered_lineup['artist'].unique())
                
                # Overlap berekenen tussen gefilterde lineup artiesten en artist insights
                if len(filtered_lineup_artists) > 0 and 'artist' in artists.columns:
                    filtered_overlap = filtered_lineup_artists.intersection(insight_artists)

                    # Duidelijkere output met divider
                    print(f"\n=== Overlap van Artiesten NA Ticket Filtering ===")
                    print(f"Events vóór filtering: {before}")
                    print(f"Events na filtering: {after}")
                    print(f"Artiesten vóór filtering: {len(lineup_artists)}")
                    print(f"Artiesten na filtering: {len(filtered_lineup_artists)}")
                    print(f"Overlap met insights vóór filtering: {len(overlap)} van {len(lineup_artists)} ({len(overlap)/len(lineup_artists)*100:.1f}%)")
                    print(f"Overlap met insights na filtering: {len(filtered_overlap)} van {len(filtered_lineup_artists)} ({len(filtered_overlap)/len(filtered_lineup_artists)*100:.1f}%)")

                    # Voeg extra informatieve details toe over artiesten die wegvallen
                    if len(filtered_lineup_artists) < len(lineup_artists):
                        lost_artists_count = len(lineup_artists) - len(filtered_lineup_artists)
                        lost_artists_pct = lost_artists_count / len(lineup_artists) * 100
                        print(f"Na ticket filtering zijn {lost_artists_count} artiesten uit line-up verwijderd ({lost_artists_pct:.1f}%)")
                        
                        # Bereken welke artiesten met insights verdwijnen na filtering
                        lost_artists_with_insights = (lineup_artists.intersection(insight_artists)) - (filtered_lineup_artists.intersection(insight_artists))
                        if lost_artists_with_insights:
                            print(f"Aantal verwijderde artiesten met insights: {len(lost_artists_with_insights)} van {len(overlap)} ({len(lost_artists_with_insights)/len(overlap)*100:.1f}%)")
                            
                    print("========================================================")

        if df_features.empty:
            if verbose:
                print(f"Geen data over na feature engineering en NaN drop voor T={T}. Overslaan.")
            continue

        # Run model comparison
        summary_df, best_model_name, best_model, target_transformed = run_model_comparison(df_features, T=T, verbose=verbose)

        if best_model_name and best_model:
            if verbose:
                print(f"Beste model voor T={T}: {best_model_name} (Target Transformed: {target_transformed})")
            # Sla model EN transformatie status op
            best_models[T] = (best_model, target_transformed)
        else:
            if verbose:
                print(f"Geen geschikt model gevonden voor T={T}")
    
    return best_models

if __name__ == "__main__":
    # --- Configuratie ---
    tickets_path = 'tickets_processed.csv'
    line_up_path = 'line_up_processed_new.csv'
    artists_path = 'artist_insights.csv'
    future_events_path = 'tickets_processed.csv' 
    T_values_to_run = [14]  # Lijst van T-waardes om te trainen
    run_verbose = True  # Zet op TRUE voor details, plots, importance etc.

    # --- Training ---
    print("=== TICKET PREDICTION MODEL TRAINING ===")

    # Roep main aan om data te laden en model comparison uit te voeren
    trained_models_dict = main(
        tickets_path=tickets_path,
        line_up_path=line_up_path,
        artists_path=artists_path,
        T_values=T_values_to_run,
        verbose=run_verbose 
    )

    # Loop door alle getrainde modellen
    for T_value in T_values_to_run:
        print(f"\n{'='*60}")
        print(f"VERWERKING RESULTATEN VOOR T={T_value}")
        print(f"{'='*60}")
        
        # Controleer of training succesvol was voor deze T
        if T_value in best_models and best_models[T_value] is not None and best_models[T_value][0] is not None:
            print(f"\n--- Training en Analyse voor T={T_value} voltooid ---")
            best_model_pipeline, target_transformed_flag = best_models[T_value]
            
            # Bepaal model naam
            best_model_reg_name = "UnknownModel"
            if isinstance(best_model_pipeline, dict) and best_model_pipeline.get('type') == 'ensemble':
                best_model_reg_name = "Ensemble"
            else:
                try:
                    best_model_reg_name = best_model_pipeline.named_steps['reg'].__class__.__name__
                except Exception:
                    pass
            
            print(f"   Beste model: {best_model_reg_name}")
            print(f"   Target was getransformeerd: {target_transformed_flag}")

            # --- Model Opslaan ---
            model_filename = f"ticket_prediction_model_T{T_value}_CV_{best_model_reg_name}.pkl"

            if save_model(model_filename):
                print(f"   Model opgeslagen als: {model_filename}")
                
                # Test laden van dit specifieke model
                # Bewaar huidige best_models
                temp_best_models = best_models.copy()
                best_models.clear()
                
                if load_model(model_filename):
                    if T_value in best_models:
                        loaded_model_pipeline, loaded_target_transformed = best_models[T_value]
                        
                        # Bepaal geladen model naam voor verificatie
                        loaded_model_name = "UnknownModel"
                        if isinstance(loaded_model_pipeline, dict) and loaded_model_pipeline.get('type') == 'ensemble':
                            loaded_model_name = "Ensemble"
                        else:
                            try:
                                loaded_model_name = loaded_model_pipeline.named_steps['reg'].__class__.__name__
                            except Exception:
                                pass
                        
                        print(f"   Model succesvol geladen en geverifieerd: {loaded_model_name}")

                        # --- Voorspellingen Doen ---
                        print(f"\n--- Start Voorspellingen (T={T_value}) met model: {loaded_model_name} ---")
                        
                        # Eerst het tickets dataframe inladen
                        future_events, line_up_df, artists_df = load_data(
                            future_events_path, line_up_path, artists_path, verbose=False
                        )
                        
                        # Juiste parameternamen gebruiken
                        known_cities = get_real_cities_present_in_data(future_events, verbose=False)
                        
                        predictions = predict_tickets(
                            df_events=future_events,
                            T=T_value,
                            known_cities=known_cities,
                            line_up_df=line_up_df,
                            artists_df=artists_df,
                            verbose=False  # Zet op True voor gedetailleerde predictie output
                        )
                        
                        # Herstel originele best_models voor volgende iteratie
                        best_models.clear()
                        best_models.update(temp_best_models)
                        
                        if not predictions.empty:
                            print(f"\n--- Voorbeeld Voorspellingen (T={T_value}) ---")
                            print(predictions.head().to_string())
                        else:
                            print("Genereren van voorspellingen is mislukt.")
                    else:
                        print(f"Fout: Model voor T={T_value} niet gevonden na laden.")
                else:
                    print(f"Laden van model {model_filename} mislukt.")
            else:
                print(f"Opslaan van model {model_filename} mislukt.")
        else:
            print(f"Training voor T={T_value} is mislukt of heeft geen model opgeleverd.")

print("\n--- Script Voltooid ---") 