import pickle
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple

# Import the global best_models dictionary
from ticket_prediction.config.constants import best_models

def save_model(model_path: str = "ticket_prediction_model.pkl") -> bool:
    """
    Sla de beste modellen (met transformatie status) op naar een bestand.
    """
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(best_models, f)
        print(f"Model(len) succesvol opgeslagen naar {model_path}")
        return True
    except Exception as e:
        print(f"Fout bij het opslaan van het model: {e}")
        return False

def load_model(model_path: str = "ticket_prediction_model.pkl") -> bool:
    """
    Laad modellen (met transformatie status) uit een bestand.
    """
    try:
        import pickle
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
            # Validatie van geladen data (optioneel maar aanbevolen)
            if isinstance(loaded_data, dict):
                # Check of de waarden tuples zijn (Pipeline of dict, bool)
                valid = True
                for key, value in loaded_data.items():
                    # Accepteer zowel Pipeline objecten als dictionaries (voor ensembles)
                    is_valid_model_type = isinstance(value[0], (Pipeline, dict))
                    
                    if not (isinstance(key, int) and isinstance(value, tuple) and len(value) == 2 and is_valid_model_type and isinstance(value[1], bool)):
                        valid = False
                        print(f"Waarschuwing: Ongeldig formaat gevonden voor key {key} in geladen modelbestand.")
                        break
                if valid:
                    # Update de bestaande dictionary in plaats van te reassignen
                    best_models.clear()
                    best_models.update(loaded_data)
                    print(f"Model(len) succesvol geladen van {model_path}. Keys: {list(best_models.keys())}")
                    return True
                else:
                    print(f"Fout: Geladen bestand {model_path} heeft niet de verwachte structuur.")
                    return False
            else:
                 print(f"Fout: Geladen bestand {model_path} is geen dictionary.")
                 return False
    except FileNotFoundError:
        print(f"Fout: Modelbestand {model_path} niet gevonden.")
        return False
    except Exception as e:
        print(f"Fout bij het laden van het model: {e}")
        return False 