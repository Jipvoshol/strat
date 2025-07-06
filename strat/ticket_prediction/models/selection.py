import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFECV, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from typing import List, Tuple, Optional, Any

def select_features_rfecv(
    X: pd.DataFrame,
    y: pd.Series,
    verbose: bool = True
) -> List[str]:
    """
    Feature selectie met RFECV.
    
    Parameters:
        X: Feature DataFrame
        y: Target Series
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        List met geselecteerde feature namen
    """
    if verbose:
        print("\n--- Poging: RFECV met een basis model ---")
    
    try:
        X_rfecv = X.copy()
        
        # Converteer categorische features naar numeriek
        cat_cols_rfecv = X_rfecv.select_dtypes(include=['category', 'object']).columns.tolist()
        if cat_cols_rfecv:
            if verbose:
                print(f"   Converteren van {len(cat_cols_rfecv)} categorische features voor RFECV...")
            X_rfecv = pd.get_dummies(X_rfecv, columns=cat_cols_rfecv, drop_first=True)
        
        # Zorg ervoor dat alles numeriek is
        X_rfecv = X_rfecv.select_dtypes(include=np.number)
        
        # Gebruik een simpel model voor RFECV
        base_model = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )
        
        # Configureer RFECV
        rfecv = RFECV(
            estimator=base_model,
            step=1,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_root_mean_squared_error',
            min_features_to_select=8,
            n_jobs=-1
        )
        
        if verbose:
            print("   Start RFECV fitting (dit kan even duren)...")
        rfecv.fit(X_rfecv, y)
        
        # Bepaal geselecteerde features
        support_mask_rfecv = rfecv.support_
        rfecv_features = X_rfecv.columns[support_mask_rfecv].tolist()
        
        # Map one-hot encoded features terug naar oorspronkelijke categorische features
        if cat_cols_rfecv:
            selected_cat_features = set()
            for feature in rfecv_features:
                for cat_col in cat_cols_rfecv:
                    if feature.startswith(f"{cat_col}_"):
                        selected_cat_features.add(cat_col)
                        break
            
            orig_features_selected = [f for f in X.columns if f in rfecv_features or f in selected_cat_features]
        else:
            orig_features_selected = [f for f in X.columns if f in rfecv_features]
        
        if verbose:
            print(f"   RFECV succesvol: {len(orig_features_selected)} features geselecteerd.")
            print(f"   Optimaal aantal features: {rfecv.n_features_}")
        
        return orig_features_selected
    
    except Exception as e:
        if verbose:
            print(f"   FOUT tijdens RFECV: {e}")
        return []

def select_features_tree_based(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Optional[Any] = None,
    verbose: bool = True
) -> List[str]:
    """
    Feature selectie met SelectFromModel en XGBoost.
    
    Parameters:
        X: Feature DataFrame
        y: Target Series
        preprocessor: Optionele preprocessor pipeline
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        List met geselecteerde feature namen
    """
    if verbose:
        print("\n--- Poging: SelectFromModel met XGBoost ---")
    
    try:
        # Als preprocessor gegeven is, gebruik deze
        if preprocessor is not None:
            preprocessor.fit(X, y)
            X_processed = preprocessor.transform(X)
        else:
            X_processed = X.values
        
        # Train een XGBoost model
        xgb_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )
        xgb_model.fit(X_processed, y)
        
        # Gebruik SelectFromModel
        sfm = SelectFromModel(
            xgb_model, 
            threshold='mean',
            prefit=True
        )
        
        # Bepaal geselecteerde features
        support_mask = sfm.get_support()
        
        if preprocessor is not None:
            # Map terug naar originele feature namen
            processed_feature_names = preprocessor.get_feature_names_out()
            selected_processed_features = processed_feature_names[support_mask]
            
            original_features_selected = set()
            for processed_name in selected_processed_features:
                if processed_name.startswith('num__'):
                    original_features_selected.add(processed_name[len('num__'):])
                elif processed_name.startswith('cat__'):
                    original_name = processed_name.split('__')[1].rsplit('_', 1)[0]
                    original_features_selected.add(original_name)
            
            selected_features = [f for f in X.columns if f in original_features_selected]
        else:
            selected_features = X.columns[support_mask].tolist()
        
        if verbose:
            print(f"   SelectFromModel succesvol: {len(selected_features)} features geselecteerd.")
        
        return selected_features
    
    except Exception as e:
        if verbose:
            print(f"   FOUT tijdens SelectFromModel: {e}")
        return []

def select_features_hybrid(
    X: pd.DataFrame,
    y: pd.Series,
    k_best: int = 40,
    n_features: int = 25,
    verbose: bool = True
) -> List[str]:
    """
    Hybride feature selectie met SelectKBest en Random Forest.
    
    Parameters:
        X: Feature DataFrame
        y: Target Series
        k_best: Aantal features voor SelectKBest
        n_features: Uiteindelijk aantal features om te selecteren
        verbose: Of debug berichten geprint moeten worden
        
    Returns:
        List met geselecteerde feature namen
    """
    if verbose:
        print("\n--- Poging: Hybride Feature Selectie (SelectKBest + Tree-based) ---")
    
    try:
        X_hybrid = X.copy()
        
        # Categoriaal naar numeriek converteren
        cat_cols = X_hybrid.select_dtypes(include=['category', 'object']).columns.tolist()
        if cat_cols:
            X_hybrid = pd.get_dummies(X_hybrid, columns=cat_cols, drop_first=True)
        
        # Zorg ervoor dat alles numeriek is
        X_hybrid = X_hybrid.select_dtypes(include=np.number)
        
        # STAP 1: SelectKBest
        k_best = min(k_best, X_hybrid.shape[1])
        skbest = SelectKBest(f_regression, k=k_best)
        skbest.fit(X_hybrid, y)
        selected_mask_skbest = skbest.get_support()
        selected_features_skbest = X_hybrid.columns[selected_mask_skbest].tolist()
        
        if verbose:
            print(f"   SelectKBest selecteerde {len(selected_features_skbest)} features.")
        
        # STAP 2: RandomForest voor feature importance
        rf_selector = RandomForestRegressor(
            n_estimators=100, 
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit op de subset van features
        X_subset = X_hybrid[selected_features_skbest]
        rf_selector.fit(X_subset, y)
        
        # Verkrijg feature importances
        feature_importances = rf_selector.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features_skbest,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Selecteer de top-N meest belangrijke features
        n_features = min(n_features, len(feature_importance_df))
        final_selected_features = feature_importance_df.head(n_features)['Feature'].tolist()
        
        # Map features terug naar originele kolommen
        if cat_cols:
            orig_selected_features = []
            for feature in final_selected_features:
                # Check of dit een one-hot encoded feature is
                for cat_col in cat_cols:
                    if feature.startswith(f"{cat_col}_"):
                        if cat_col not in orig_selected_features:
                            orig_selected_features.append(cat_col)
                        break
                else:  # Dit is een reguliere numerieke feature
                    orig_selected_features.append(feature)
            
            selected_feature_names = [f for f in X.columns if f in orig_selected_features]
        else:
            selected_feature_names = [f for f in X.columns if f in final_selected_features]
        
        if verbose:
            print(f"   Hybride selectie succesvol: {len(selected_feature_names)} features geselecteerd.")
        
        return selected_feature_names
    
    except Exception as e:
        if verbose:
            print(f"   FOUT tijdens Hybride Feature Selectie: {e}")
        return []

def get_default_features() -> List[str]:
    """
    Geeft een lijst met default features terug als fallback.
    
    Returns:
        List met feature namen
    """
    return [
        'log_cumulative_sales_at_t',
        'log_SpotifyFollowers_max', 
        'log_ChartmetricScore_max',
        'log_sales_last_3_days',
        'log_avg_daily_sales_before_t',
        'avg_product_value',
        'day_of_week', 
        'is_weekend',
        'month_cos', 
        't_max', 
        'rain_fall',
        'log_unique_cities', 
        'log_max_capacity'
    ] 