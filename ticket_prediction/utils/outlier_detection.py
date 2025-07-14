from typing import List
import pandas as pd

def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'IQR', factor: float = 1.5, verbose: bool = True) -> pd.DataFrame:
    """
    Verbeterde versie: Verwijdert outliers uit de opgegeven kolommen met meer robuuste methoden.

    Parameters:
        df (pd.DataFrame): Invoer DataFrame.
        columns (List[str]): Lijst van kolomnamen om outliers uit te filteren.
        method (str): Methode om outliers te detecteren: 'IQR', 'zscore', of 'combined'.
        factor (float): De vermenigvuldigingsfactor voor de IQR of het aantal standaarddeviaties.
        verbose (bool): Indien True worden statusberichten getoond.

    Returns:
        pd.DataFrame: DataFrame zonder outliers.
    """
    result_df = df.copy()
    total_outliers = 0
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        # Haal alleen numerieke waarden op voor berekening
        valid_values = result_df[col].dropna()
        if len(valid_values) == 0:
            continue
            
        if method == 'IQR' or method == 'combined':
            # IQR methode (robuuster tegen uitschieters dan z-score)
            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound_iqr = Q1 - factor * IQR
            upper_bound_iqr = Q3 + factor * IQR
            
            if method == 'IQR':
                before = result_df.shape[0]
                result_df = result_df[((result_df[col] >= lower_bound_iqr) & (result_df[col] <= upper_bound_iqr)) | (result_df[col].isna())]
                after = result_df.shape[0]
                removed = before - after
                total_outliers += removed
        
        if method == 'zscore' or method == 'combined':
            # Z-score methode
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            if std_val > 0:  # Voorkom delen door nul
                z_scores = (result_df[col] - mean_val) / std_val
                lower_bound_z = -factor
                upper_bound_z = factor
                
                if method == 'zscore':
                    before = result_df.shape[0]
                    result_df = result_df[((z_scores >= lower_bound_z) & (z_scores <= upper_bound_z)) | z_scores.isna()]
                    after = result_df.shape[0]
                    removed = before - after
                    total_outliers += removed
        
        if method == 'combined':
            # Combineer beide methoden (striktere filtering)
            before = result_df.shape[0]
            mask_iqr = ((result_df[col] >= lower_bound_iqr) & (result_df[col] <= upper_bound_iqr)) | result_df[col].isna()
            if std_val > 0:
                z_scores = (result_df[col] - result_df[col].mean()) / result_df[col].std()
                mask_z = ((z_scores >= -factor) & (z_scores <= factor)) | z_scores.isna()
                mask_combined = mask_iqr & mask_z
            else:
                mask_combined = mask_iqr
            result_df = result_df[mask_combined]
            after = result_df.shape[0]
            removed = before - after
            total_outliers += removed
            
        if verbose and method in ['IQR', 'zscore', 'combined']:
            if removed > 0:
                print(f"Outliers verwijderd uit {col} met {method}: {removed} rijen ({removed/before*100:.1f}%).")
    
    if verbose and total_outliers > 0:
        print(f"Totaal {total_outliers} outliers verwijderd uit alle kolommen.")
        
    return result_df 