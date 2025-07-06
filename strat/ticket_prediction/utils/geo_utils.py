from typing import List
import pandas as pd
import geonamescache

def get_real_cities_present_in_data(tickets: pd.DataFrame, verbose: bool = True) -> List[str]:
    """
    Retourneert een lijst met unieke steden die zowel in de tickets als in geonamescache voorkomen.

    Parameters:
        tickets (pd.DataFrame): DataFrame met een kolom 'city'.
        verbose (bool): Indien True worden statusberichten getoond.

    Returns:
        List[str]: Lijst met relevante steden.
    """
    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()
    data_cities = set(tickets['city'].astype(str).str.lower().unique())
    real_cities = {city_info['name'].lower() for city_info in cities.values() if city_info['name'].lower() in data_cities}
    if verbose:
        print(f"Aantal relevante steden gevonden via geonamescache: {len(real_cities)}")
    return list(real_cities)

def standardize_city_names(tickets: pd.DataFrame, known_cities: List[str], verbose: bool = True) -> pd.DataFrame:
    """
    Standaardiseert de 'city'-kolom zodat namen overeenkomen met een lijst van bekende steden.
    Onbekende steden worden gemarkeerd als 'other'.

    Parameters:
        tickets (pd.DataFrame): DataFrame met een 'city'-kolom.
        known_cities (List[str]): Lijst met bekende stadennamen.
        verbose (bool): Indien True worden statusberichten getoond.

    Returns:
        pd.DataFrame: De aangepaste DataFrame met een extra kolom 'city_standardized'.
    """
    tickets['city'] = (
        tickets['city']
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    tickets['city_standardized'] = tickets['city'].apply(lambda x: x if x in known_cities else 'other')
    if verbose:
        unmatched = (tickets['city_standardized'] == 'other').sum()
        total = tickets.shape[0]
        print(f"Aantal steden gemarkeerd als 'other': {unmatched} / {total} ({unmatched/total*100:.1f}%)")
    return tickets 