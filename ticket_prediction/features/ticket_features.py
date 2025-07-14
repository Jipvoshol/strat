import pandas as pd

def engineer_ticket_features(
    tickets_upto_t: pd.DataFrame,
    df_merged: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert ticket info features.
    
    Parameters:
        tickets_upto_t: Tickets tot aan tijdstip T
        df_merged: Gemergde DataFrame tot nu toe
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        DataFrame met ticket features toegevoegd
    """
    if verbose:
        print("[engineer_ticket_features] Stap 7: Ticket Info Features...")
    
    # Start met events in df_merged
    ticket_agg = pd.DataFrame({'event_name': df_merged['event_name'].unique()})
    
    # Definieer kolommen die gecreëerd worden
    ticket_cols_created = ['avg_product_value', 'std_product_value', 'avg_total_price', 'scanned_percentage']
    required_ticket_cols = {'product_value', 'total_price', 'product_is_scanned'}
    
    # Check of we de vereiste kolommen hebben
    if required_ticket_cols.issubset(tickets_upto_t.columns) and not tickets_upto_t.empty:
        # Zorg ervoor dat de kolommen numeriek zijn voordat we aggregeren
        tickets_upto_t_numeric = tickets_upto_t.copy()
        
        # Convert product_value to numeric
        tickets_upto_t_numeric['product_value'] = pd.to_numeric(
            tickets_upto_t_numeric['product_value'], errors='coerce'
        )
        
        # Convert total_price to numeric
        tickets_upto_t_numeric['total_price'] = pd.to_numeric(
            tickets_upto_t_numeric['total_price'], errors='coerce'
        )
        
        # Convert product_is_scanned to numeric (assuming it's boolean-like or 0/1)
        tickets_upto_t_numeric['product_is_scanned'] = pd.to_numeric(
            tickets_upto_t_numeric['product_is_scanned'], errors='coerce'
        )
        
        # Now perform the aggregation with numeric columns
        df_agg_ticket = tickets_upto_t_numeric.groupby('event_name').agg(
            avg_product_value=('product_value', 'mean'),
            std_product_value=('product_value', 'std'),
            avg_total_price=('total_price', 'mean'),
            scanned_percentage=('product_is_scanned', 'mean')
        ).reset_index()
        
        ticket_agg = pd.merge(ticket_agg, df_agg_ticket, on='event_name', how='left')
        
        # Vul NaNs (bv met 0 of mediaan)
        for col in ticket_cols_created:
            if col in ticket_agg.columns:
                fill_value_ticket = ticket_agg[col].median() if pd.notna(ticket_agg[col].median()) else 0
                ticket_agg[col] = ticket_agg[col].fillna(fill_value_ticket)
            else:
                ticket_agg[col] = 0  # Maak kolom aan met 0
    else:
        if verbose:
            print("Onvoldoende data voor ticket features.")
        # Vul alle defaults met 0
        for col in ticket_cols_created:
            ticket_agg[col] = 0
    
    # Merge ticket features
    df_merged = pd.merge(df_merged, ticket_agg, on='event_name', how='left')
    
    return df_merged 