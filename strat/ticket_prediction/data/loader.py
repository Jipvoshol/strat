import pandas as pd
from typing import Tuple

def load_data(tickets_path: str, line_up_path: str, artists_path: str, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Laadt de datasets van de gegeven paden.
    """
    try:
        tickets = pd.read_csv(tickets_path, low_memory=False)
        line_up = pd.read_csv(line_up_path, low_memory=False)
        artists = pd.read_csv(artists_path, low_memory=False)
        if verbose:
            print(f"CSV bestanden geladen:\n- {tickets_path}: {tickets.shape}\n- {line_up_path}: {line_up.shape}\n- {artists_path}: {artists.shape}")
        return tickets, line_up, artists
    except FileNotFoundError as e:
        print(f"Fout: Bestand niet gevonden - {e}")
        raise
    except Exception as e:
        print(f"Fout bij het laden van de data: {e}")
        raise 