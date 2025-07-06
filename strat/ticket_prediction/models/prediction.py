import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from sklearn.pipeline import Pipeline

from ticket_prediction.features.engineering import engineer_features
from ticket_prediction.config.constants import best_models

def predict_tickets(
    df_events: pd.DataFrame,
    T: int,
    known_cities: Optional[List[str]],
    line_up_df: pd.DataFrame,
    artists_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Voorspel het totaal aantal tickets voor één of meerdere evenementen.
    Past nu np.expm1 toe indien nodig.
    
    Parameters:
        df_events: DataFrame met event data
        T: Forecast horizon in dagen
        known_cities: Lijst met bekende steden
        line_up_df: Line-up DataFrame
        artists_df: Artists DataFrame
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        DataFrame met voorspellingen
    """
    if T not in best_models:
        if verbose:
            print(f"Geen model beschikbaar voor T={T} dagen. Zorg ervoor dat het model voor deze T is getraind.")
        return pd.DataFrame()

    # Haal model en transformatie flag op
    model_info = best_models[T]
    if not isinstance(model_info, tuple) or len(model_info) != 2:
        if verbose:
            print(f"Fout: Ongeldige model informatie opgeslagen voor T={T}. Verwachtte (Pipeline, bool).")
        return pd.DataFrame()
    
    model, target_transformed = model_info

    if model is None:
        if verbose:
            print(f"Fout: Model voor T={T} is None.")
        return pd.DataFrame()

    # Check if model is an ensemble
    is_ensemble = isinstance(model, dict) and model.get('type') == 'ensemble'
    
    # Engineer features for prediction
    df_features = engineer_features(
        tickets=df_events.copy(),
        line_up_df=line_up_df.copy() if line_up_df is not None else pd.DataFrame(),
        artists_df=artists_df.copy() if artists_df is not None else pd.DataFrame(),
        forecast_days=T,
        known_cities=known_cities,
        max_lag=3,
        is_prediction=True,
        verbose=False  # Niet verbose tijdens predictie
    )

    if df_features.empty:
        if verbose:
            print("Feature engineering leverde een leeg dataframe op voor predictie.")
        return pd.DataFrame()

    # Get required features based on model
    try:
        if is_ensemble:
            # Voor ensemble models, gebruik eerste base model om features te bepalen
            first_base_model = list(model['base_models'].values())[0]
            if hasattr(first_base_model, 'named_steps'):
                preprocessor = first_base_model.named_steps['preprocessor']
            else:
                # Direct getraind model zonder pipeline
                if hasattr(first_base_model, 'feature_names_in_'):
                    required_features = first_base_model.feature_names_in_
                else:
                    if verbose:
                        print("WAARSCHUWING: Kan features niet bepalen voor ensemble model.")
                    required_features = df_features.columns.tolist()
        else:
            # Regular pipeline model
            preprocessor = model.named_steps['preprocessor']
            num_features = preprocessor.transformers_[0][2]  # Numerieke features
            cat_features = preprocessor.transformers_[1][2]  # Categorische features
            required_features = list(num_features) + list(cat_features)
    except Exception as e:
        if verbose:
            print(f"Fout bij ophalen vereiste features uit model: {e}")
        # Fallback: probeer alle kolommen te gebruiken
        try:
            if hasattr(model, 'named_steps') and hasattr(model.named_steps['reg'], 'feature_names_in_'):
                required_features = model.named_steps['reg'].feature_names_in_
            else:
                required_features = df_features.columns.tolist()
                if verbose:
                    print("WAARSCHUWING: Gebruik alle beschikbare features als fallback.")
        except Exception:
            if verbose:
                print("FATAL: Kan vereiste features niet bepalen.")
            return pd.DataFrame()

    # Controleer of alle vereiste features aanwezig zijn
    missing_features = [f for f in required_features if f not in df_features.columns]
    if missing_features:
        if verbose:
            print(f"Waarschuwing: Deze features ontbreken in de input data voor predictie: {missing_features}")
        # Vul ontbrekende features in met 0
        for feature in missing_features:
            df_features[feature] = 0
            if verbose:
                print(f"   -> Feature '{feature}' aangevuld met 0.")

    # Selecteer alleen de benodigde features in de juiste volgorde
    try:
        X_pred = df_features[required_features]
    except KeyError as e:
        if verbose:
            print(f"Fout bij selecteren features voor predictie: {e}. Zorg dat alle benodigde kolommen bestaan.")
        return pd.DataFrame()

    # Voorspel
    try:
        if is_ensemble:
            # Ensemble prediction
            predictions_raw = predict_with_ensemble(model, X_pred)
        else:
            # Regular model prediction
            predictions_raw = model.predict(X_pred)

        # --- TERUGTRANSFORMATIE ---
        if target_transformed:
            if verbose:
                print("Terugtransformeren van log-voorspellingen met np.expm1().")
            predictions_orig_scale = np.expm1(predictions_raw)
            # Zorg dat voorspellingen niet negatief zijn
            predictions_orig_scale[predictions_orig_scale < 0] = 0
        else:
            predictions_orig_scale = predictions_raw

        df_features['predicted_full_event_tickets'] = predictions_orig_scale

        # Rond voorspellingen af op gehele getallen
        df_features['predicted_full_event_tickets'] = np.round(
            df_features['predicted_full_event_tickets']
        ).astype(int)

        # Toon voorspellingen (optioneel)
        if verbose:
            for idx, row in df_features.iterrows():
                event_name = row['event_name']
                predicted = row['predicted_full_event_tickets']
                actual = row.get('full_event_tickets', 'onbekend')

                if actual != 'onbekend':
                    print(f"{event_name} - voorspeld: {predicted}, werkelijk: {actual}")
                else:
                    print(f"{event_name} - voorspeld: {predicted}")

        # Return alleen relevante kolommen
        return df_features[['event_name', 'predicted_full_event_tickets']]

    except Exception as e:
        if verbose:
            print(f"Fout bij voorspellen of terugtransformeren: {e}")
        return pd.DataFrame()

def predict_with_ensemble(ensemble_model: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """
    Maakt voorspellingen met een ensemble model.
    
    Parameters:
        ensemble_model: Dictionary met ensemble model componenten
        X: Feature DataFrame
        
    Returns:
        Array met voorspellingen
    """
    base_models = ensemble_model['base_models']
    meta_learner = ensemble_model['meta_learner']
    
    # Genereer base model predictions
    base_predictions = []
    for name, model in base_models.items():
        pred = model.predict(X)
        base_predictions.append(pred)
    
    # Stack predictions als features voor meta learner
    meta_features = np.column_stack(base_predictions)
    
    # Meta learner prediction
    final_predictions = meta_learner.predict(meta_features)
    
    return final_predictions

def get_closest_trained_model_T(requested_T: int) -> int:
    """
    Vindt de dichtstbijzijnde T waarvoor een model is getraind.
    
    Parameters:
        requested_T: Gevraagde forecast horizon
        
    Returns:
        int: Dichtstbijzijnde T waarvoor een model beschikbaar is
    """
    available_Ts = sorted(list(best_models.keys()))
    if not available_Ts:
        raise ValueError("Geen modellen beschikbaar. Train eerst modellen.")
    
    if requested_T in available_Ts:
        return requested_T
    
    # Vind de dichtstbijzijnde T
    closest_T = min(available_Ts, key=lambda x: abs(x - requested_T))
    print(f"Geen model voor T={requested_T}, gebruikt model voor T={closest_T}")
    return closest_T

def get_future_events(tickets_df: pd.DataFrame) -> List[str]:
    """
    Haalt evenementen op die nog moeten plaatsvinden.
    
    Parameters:
        tickets_df: DataFrame met ticket data
        
    Returns:
        List met event namen die in de toekomst liggen
    """
    if 'first_event_date_start' not in tickets_df.columns:
        return []
    
    try:
        tickets_df['first_event_date_start'] = pd.to_datetime(tickets_df['first_event_date_start'])
        today = pd.Timestamp.now().normalize()
        future_events = tickets_df[
            tickets_df['first_event_date_start'] > today
        ]['event_name'].unique().tolist()
        return future_events
    except Exception as e:
        print(f"Fout bij ophalen toekomstige events: {e}")
        return [] 