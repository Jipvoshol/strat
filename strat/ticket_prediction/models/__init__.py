"""Machine learning models and training modules"""

from ticket_prediction.models.training import run_model_comparison
from ticket_prediction.models.prediction import predict_tickets, get_future_events
from ticket_prediction.models.metrics import rmse_scorer, get_scoring_dict
from ticket_prediction.models.preprocessors import RmseOptimizedPreprocessor
from ticket_prediction.models.selection import (
    select_features_rfecv,
    select_features_tree_based, 
    select_features_hybrid,
    get_default_features
)
from ticket_prediction.models.ensemble import create_ensemble_model, ensemble_cross_predict

__all__ = [
    'run_model_comparison',
    'predict_tickets',
    'get_future_events',
    'rmse_scorer',
    'get_scoring_dict',
    'RmseOptimizedPreprocessor',
    'select_features_rfecv',
    'select_features_tree_based',
    'select_features_hybrid',
    'get_default_features',
    'create_ensemble_model',
    'ensemble_cross_predict'
]

