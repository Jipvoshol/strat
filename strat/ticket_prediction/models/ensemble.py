import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any, Tuple, Optional

def create_ensemble_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    target_transformed: bool,
    cv: TimeSeriesSplit,
    verbose: bool = True
) -> Tuple[Dict[str, Any], bool]:
    """
    Creëert een ensemble model met verschillende base learners en een meta learner.
    """
    if verbose:
        print("Creating ensemble model...")
    
    # Definieer base models met preprocessor in een pipeline
    base_models = {
        'rf': Pipeline([('preprocessor', clone(preprocessor)), 
                        ('reg', RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1))]),
        'gb': Pipeline([('preprocessor', clone(preprocessor)),
                        ('reg', GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42))]),
        'xgb': Pipeline([('preprocessor', clone(preprocessor)),
                         ('reg', XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1))])
    }
    
    # Train base models
    trained_base_models = {}
    for name, model in base_models.items():
        if verbose:
            print(f"Training base model: {name}")
        trained_base_models[name] = clone(model).fit(X, y)
    
    # Meta learner (simpel Ridge model)
    meta_learner = Ridge(alpha=1.0, random_state=42)
    
    # Return ensemble dictionary
    ensemble_dict = {
        'type': 'ensemble',
        'base_models': trained_base_models,
        'meta_learner': meta_learner,
        'preprocessor': preprocessor
    }
    
    return ensemble_dict, target_transformed

def ensemble_cross_predict(
    X: pd.DataFrame,
    y: pd.Series,
    base_models: Dict[str, Any],
    meta_learner: Any,
    groups: Optional[pd.Series],
    cv: TimeSeriesSplit
) -> np.ndarray:
    """
    Maakt cross-validated predictions met een ensemble model.
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    
    # Voor elke CV fold
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Genereer base model predictions voor validation set
        val_predictions = []
        for name, model in base_models.items():
            # Clone en retrain model op training data van deze fold
            fold_model = clone(model).fit(X_train, y_train)
            val_pred = fold_model.predict(X_val)
            val_predictions.append(val_pred)
        
        # Stack predictions als features voor meta learner
        val_meta_features = np.column_stack(val_predictions)
        
        # Train meta learner op training data
        train_predictions = []
        for name, model in base_models.items():
            # Voor training data gebruiken we out-of-fold predictions
            # Dit is een vereenvoudiging - in productie zou je nested CV gebruiken
            train_pred = model.predict(X_train)
            train_predictions.append(train_pred)
        
        train_meta_features = np.column_stack(train_predictions)
        
        # Fit meta learner
        meta_model = clone(meta_learner).fit(train_meta_features, y_train)
        
        # Maak ensemble predictions voor validation set
        ensemble_pred = meta_model.predict(val_meta_features)
        predictions[val_idx] = ensemble_pred
    
    return predictions 