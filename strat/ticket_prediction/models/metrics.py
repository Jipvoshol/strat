import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

def rmse_scorer(y_true, y_pred) -> float:
    """
    Berekent Root Mean Squared Error tussen werkelijke en voorspelde waarden.
    
    Parameters:
        y_true: Werkelijke waarden
        y_pred: Voorspelde waarden
        
    Returns:
        float: RMSE waarde
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmse_original_scorer(y_true_log, y_pred_log):
    """RMSE scorer voor log-getransformeerde data, terug getransformeerd naar originele schaal."""
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    y_pred_orig[y_pred_orig < 0] = 0
    # Check for NaN/inf after expm1 before calculating metric
    valid_mask = np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)
    if valid_mask.sum() == 0: 
        return np.inf
    return rmse_scorer(y_true_orig[valid_mask], y_pred_orig[valid_mask])

def mae_original_scorer(y_true_log, y_pred_log):
    """MAE scorer voor log-getransformeerde data, terug getransformeerd naar originele schaal."""
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    y_pred_orig[y_pred_orig < 0] = 0
    valid_mask = np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)
    if valid_mask.sum() == 0: 
        return np.inf
    return mean_absolute_error(y_true_orig[valid_mask], y_pred_orig[valid_mask])

def r2_original_scorer(y_true_log, y_pred_log):
    """R2 scorer voor log-getransformeerde data, terug getransformeerd naar originele schaal."""
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    y_pred_orig[y_pred_orig < 0] = 0
    valid_mask = np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)
    if valid_mask.sum() < 2: 
        return -np.inf  # R2 needs at least 2 samples
    return r2_score(y_true_orig[valid_mask], y_pred_orig[valid_mask])

def get_scoring_dict(target_transformed: bool = False):
    """
    Geeft een dictionary met scorers terug.
    
    Parameters:
        target_transformed: Of de target variabele getransformeerd is (log)
        
    Returns:
        dict: Dictionary met scorers
    """
    if target_transformed:
        return {
            'r2_orig': make_scorer(r2_original_scorer, greater_is_better=True),
            'neg_rmse_orig': make_scorer(rmse_original_scorer, greater_is_better=False),
            'neg_mae_orig': make_scorer(mae_original_scorer, greater_is_better=False)
        }
    else:
        return {
            'r2': 'r2',
            'neg_rmse': make_scorer(rmse_scorer, greater_is_better=False),
            'neg_mae': 'neg_mean_absolute_error'
        } 