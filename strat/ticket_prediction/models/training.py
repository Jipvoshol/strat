import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ticket_prediction.models.preprocessors import RmseOptimizedPreprocessor
from ticket_prediction.models.metrics import get_scoring_dict, rmse_scorer
from ticket_prediction.models.selection import (
    select_features_rfecv, select_features_tree_based, 
    select_features_hybrid, get_default_features
)
from ticket_prediction.models.ensemble import create_ensemble_model, ensemble_cross_predict
from ticket_prediction.config.constants import RANDOM_STATE

# Suppress warnings
warnings.filterwarnings('ignore')

def get_model_configs(preprocessor):
    """
    Geeft model configuraties terug.
    
    Parameters:
        preprocessor: Preprocessing pipeline
        
    Returns:
        Dictionary met model configuraties
    """
    models = {
        "Lasso": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("reg", Lasso(random_state=RANDOM_STATE, max_iter=10_000_000, tol=1e-4)),
            ]),
            "search_type": "grid",
            "params": {"reg__alpha": np.logspace(-5, 0, 20)},
        },
        "Ridge": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("reg", Ridge(random_state=RANDOM_STATE, max_iter=10_000_000, tol=1e-4)),
            ]),
            "search_type": "grid",
            "params": {"reg__alpha": np.logspace(-3, 3, 15)},
        },
        "ElasticNet": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("reg", ElasticNet(random_state=RANDOM_STATE, max_iter=10_000_000, tol=1e-4)),
            ]),
            "search_type": "grid",
            "params": {
                "reg__alpha": np.logspace(-4, -1, 10),
                "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
            },
        },
        "RandomForest": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("reg", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            "search_type": "random",
            "params": {
                "reg__n_estimators": [200, 400],
                "reg__max_depth": [4, 5, 6],
                "reg__min_samples_leaf": [5, 10],
                "reg__min_samples_split": [10, 20],
                "reg__max_features": ["sqrt", 0.7],
            },
        },
        "GradientBoost": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("reg", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]),
            "search_type": "random",
            "params": {
                "reg__n_estimators": [100, 150, 200],
                "reg__max_depth": [2, 3],
                "reg__min_samples_leaf": [10, 15],
                "reg__min_samples_split": [20, 30],
                "reg__learning_rate": [0.03, 0.05, 0.07],
                "reg__subsample": [0.75, 0.85],
            },
        },
        "XGBoost": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("reg", XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    enable_categorical=False,
                )),
            ]),
            "search_type": "random",
            "params": {
                "reg__learning_rate": [0.03, 0.04, 0.05],
                "reg__n_estimators": [500, 600, 700],
                "reg__max_depth": [2, 3],
                "reg__min_child_weight": [1, 3],
                "reg__subsample": [0.75, 0.80, 0.85],
                "reg__colsample_bytree": [0.85, 0.90],
                "reg__gamma": [0],
                "reg__reg_alpha": [0, 0.05, 0.10],
                "reg__reg_lambda": [0.5, 1.0],
            },
        },
        "CatBoost": {
            "model": Pipeline([
                ("preprocessor", preprocessor),
                ("reg", CatBoostRegressor(
                    random_state=RANDOM_STATE,
                    loss_function="RMSE",
                    verbose=0,
                )),
            ]),
            "search_type": "random",
            "params": {
                "reg__iterations": [400, 500, 600],
                "reg__learning_rate": [0.03, 0.05],
                "reg__depth": [4, 5],
                "reg__l2_leaf_reg": [3, 5, 7],
                "reg__subsample": [0.75, 0.85],
                "reg__min_data_in_leaf": [1, 3],
            },
            "fit_params": {},
        },
    }
    return models

def run_model_comparison(
    df: pd.DataFrame, 
    T: int = 7, 
    verbose: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Pipeline], bool]:
    """
    Voert modelvergelijking uit met:
    - Target Transformatie (Log)
    - Feature Selectie (SelectKBest met f_regression)
    - Stabiliteit-gefocuste Tuning
    - XGBoost & CatBoost toegevoegd
    - Focus op betrouwbare CV-scores en duidelijke output/visualisaties.
    - Optioneel ensemble model voor verbeterde prestaties
    
    Retourneert: summary_df, best_model_name, best_model_pipeline, target_transformed
    """
    if verbose:
        print(f"\n=== Start Geavanceerde Model Comparison (T={T}) ===")
    
    target_col = 'full_event_tickets'
    
    # --- Initiële Checks en Target Transformatie ---
    if target_col not in df.columns:
        if verbose:
            print(f"Target '{target_col}' niet gevonden.")
        return None, None, None, False
    
    df = df.dropna(subset=[target_col])
    target_transformed = False
    
    if (df[target_col] <= 0).any():
        if verbose:
            print(f"WAARSCHUWING: Target kolom '{target_col}' bevat niet-positieve waarden. Log-transformatie wordt overgeslagen.")
        y = df[target_col].copy()
        y_orig = y.copy()
    else:
        y_orig = df[target_col].copy()
        y = np.log1p(y_orig)
        target_transformed = True
        if verbose:
            print("Target variabele log-getransformeerd (np.log1p).")
    
    if df.shape[0] < 20:
        if verbose:
            print(f"Onvoldoende data na NaN drop (n={df.shape[0]} < 20).")
        return None, None, None, target_transformed
    
    # --- Feature Selectie ---
    if verbose:
        print("\n--- Start Feature Selectie ---")
    
    # Lijst van potentiële features
    all_potential_features = [
        'log_cumulative_sales_at_t',
        'log_sales_last_3_days',
        'log_avg_daily_sales_before_t',
        'avg_acceleration',
        'SpotifyPopularity_max',
        'log_SpotifyPopularity_max',
        'log_SpotifyFollowers_max',
        'SpotifyFollowers_max',
        'event_month',
        'has_star_artist', 'num_artists',
        'avg_age', 'female_percentage',
        'log_unique_cities', 'main_city_buyer_ratio',
        'event_duration_hours',
        'avg_product_value',
        't_max', 'rain_fall',
        't_min',
        'month_cos', 'day_of_week_cos', 'is_rainy'
    ]
    
    # Verwijder constante features
    removed_features = []
    for feature in ['max_capacity', 'log_max_capacity']:
        if feature in all_potential_features:
            all_potential_features.remove(feature)
            removed_features.append(feature)
    
    if removed_features and verbose:
        print(f"Verwijderde constante features uit potentiële features: {removed_features}")
    
    # Filter beschikbare features
    available_features = [f for f in all_potential_features if f in df.columns and df[f].nunique() > 1]
    
    if not available_features:
        if verbose:
            print("Geen initiële features beschikbaar na verwijderen constanten.")
        return None, None, None, target_transformed
    
    X_initial = df[available_features].copy()
    
    # Identificeer feature types
    temp_numerical = X_initial.select_dtypes(include=np.number).columns.tolist()
    temp_categorical = X_initial.select_dtypes(exclude=np.number).columns.tolist()
    potential_cats = ['day_of_week', 'event_type', 'is_weekend', 'has_star_artist', 'has_artist_data', 'is_rainy', 'event_year']
    
    for cat_col in potential_cats:
        if cat_col in X_initial.columns:
            if cat_col in temp_numerical:
                temp_numerical.remove(cat_col)
            if cat_col not in temp_categorical:
                temp_categorical.append(cat_col)
            try:
                X_initial[cat_col] = X_initial[cat_col].astype('category')
            except Exception:
                pass
    
    # Feature selectie
    selection_succeeded = False
    selected_feature_names = []
    
    # Optie 1: RFECV
    if not selection_succeeded:
        selected_feature_names = select_features_rfecv(X_initial, y, verbose)
        if selected_feature_names:
            selection_succeeded = True
            if verbose:
                print(f"RFECV succesvol: {len(selected_feature_names)} features geselecteerd.")
    
    # Optie 2: SelectFromModel
    if not selection_succeeded:
        preprocessor_temp = RmseOptimizedPreprocessor.create_pipeline(temp_numerical, temp_categorical)
        selected_feature_names = select_features_tree_based(X_initial, y, preprocessor_temp, verbose)
        if selected_feature_names:
            selection_succeeded = True
            if verbose:
                print(f"SelectFromModel succesvol: {len(selected_feature_names)} features geselecteerd.")
    
    # Optie 3: Hybride
    if not selection_succeeded:
        selected_feature_names = select_features_hybrid(X_initial, y, verbose=verbose)
        if selected_feature_names:
            selection_succeeded = True
            if verbose:
                print(f"Hybride selectie succesvol: {len(selected_feature_names)} features geselecteerd.")
    
    # Fallback: Handmatige selectie
    if not selection_succeeded:
        if verbose:
            print("\n--- Fallback: Handmatige Feature Selectie ---")
        selected_feature_names = get_default_features()
        selected_feature_names = [f for f in selected_feature_names if f in X_initial.columns]
        if not selected_feature_names:
            print("FATAL: Zelfs handmatige feature selectie levert niks op.")
            return None, None, None, target_transformed
        if verbose:
            print(f"Gebruik handmatig geselecteerde fallback features ({len(selected_feature_names)}).")
    
    X = X_initial[selected_feature_names].copy()
    
    # Update finale feature types
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Her-check categorische kolommen
    for cat_col in potential_cats:
        if cat_col in X.columns:
            if cat_col in numerical_features:
                numerical_features.remove(cat_col)
            if cat_col not in categorical_features:
                categorical_features.append(cat_col)
            try:
                if not pd.api.types.is_categorical_dtype(X[cat_col]):
                    X[cat_col] = X[cat_col].astype('category')
            except Exception:
                pass
    
    if verbose:
        print(f"\nFeatures na selectie:")
        print(f"- Numeriek ({len(numerical_features)}): {numerical_features}")
        print(f"- Categorisch ({len(categorical_features)}): {categorical_features}")
    
    if not numerical_features and not categorical_features:
        if verbose:
            print("Geen features over.")
        return None, None, None, target_transformed
    
    if verbose:
        print("--- Feature Selectie Voltooid ---")
    
    # --- TimeSeriesSplit Setup ---
    n_samples = len(X)
    
    # OPTIMALE INSTELLINGEN: n_splits = 5, test_size = n_samples // 6 (beste prestaties)
    n_splits = 5
    test_size = n_samples // 6
    
    # Zorg dat we niet meer folds hebben dan mogelijk met de dataset
    max_possible_folds = n_samples // (2 * test_size)
    n_splits = min(n_splits, max_possible_folds) if max_possible_folds > 0 else 2
    n_splits = max(2, n_splits)
    
    if verbose:
        print(f"Geoptimaliseerde TimeSeriesSplit: n_splits={n_splits}, geschatte test_size={test_size}")
        print(f"Dataset grootte: {n_samples} samples")
    
    cv = TimeSeriesSplit(n_splits=n_splits)
    
    # --- Definitieve Preprocessor ---
    try:
        preprocessor = RmseOptimizedPreprocessor.create_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
    except Exception as e:
        if verbose:
            print(f"Fout bij maken definitieve preprocessor: {e}")
        return None, None, None, target_transformed
    
    # --- Model Training ---
    models = get_model_configs(preprocessor)
    tuning_scoring = 'neg_root_mean_squared_error'
    
    results = {}
    trained_models = {}
    
    print(f"\n--- Start Model Training & CV (T={T}, Target Transformed={target_transformed}, Features={len(X.columns)}) ---")
    
    for name, config in models.items():
        if verbose:
            print(f"Training {name}...")
        
        pipeline = config['model']
        params = config['params']
        search_type = config.get('search_type', 'grid')
        search_cv_strategy = TimeSeriesSplit(n_splits=n_splits)
        
        if search_type == 'random':
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=params,
                n_iter=50,
                scoring=tuning_scoring,
                cv=search_cv_strategy,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                error_score='raise',
                verbose=0
            )
        else:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=params,
                scoring=tuning_scoring,
                cv=search_cv_strategy,
                n_jobs=-1,
                error_score='raise',
                verbose=0
            )
        
        try:
            # Do NOT use sample weights during hyperparameter tuning (like original notebook)
            search.fit(X, y)
            
            best_estimator = search.best_estimator_
            eval_cv_strategy = TimeSeriesSplit(n_splits=n_splits)
            
            eval_scoring = get_scoring_dict(target_transformed)
            
            # Use sample_weight ONLY during final cross-validation evaluation (like original notebook)
            # Handmatige cross-validation omdat fit_params problemen geeft
            cv_scores = {}
            for score_name in eval_scoring.keys():
                cv_scores[score_name] = []
            
            sample_weights = df['full_event_tickets'].loc[X.index]
            
            for train_idx, val_idx in eval_cv_strategy.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                sample_weight_fold = sample_weights.iloc[train_idx]
                
                # Clone het model voor elke fold
                from sklearn.base import clone
                fold_model = clone(best_estimator)
                
                # Fit met sample_weight
                fold_model.fit(X_train_fold, y_train_fold, reg__sample_weight=sample_weight_fold)
                
                # Voorspel op validatie set
                y_pred_fold = fold_model.predict(X_val_fold)
                
                # Bereken scores
                for score_name, scorer in eval_scoring.items():
                    try:
                        if isinstance(scorer, str):
                            # Gebruik sklearn's ingebouwde scorers
                            from sklearn.metrics import get_scorer
                            sklearn_scorer = get_scorer(scorer)
                            score = sklearn_scorer(fold_model, X_val_fold, y_val_fold)
                        else:
                            # Gebruik onze custom scorers
                            score = scorer(fold_model, X_val_fold, y_val_fold)
                        
                        cv_scores[score_name].append(score)
                    except Exception as e:
                        print(f"Fout bij berekenen score {score_name}: {e}")
                        cv_scores[score_name].append(np.nan)
            
            # Converteer naar numpy arrays
            for score_name in cv_scores:
                cv_scores[score_name] = np.array(cv_scores[score_name])
            
            r2_key = 'r2_orig' if target_transformed else 'r2'
            rmse_key = 'neg_rmse_orig' if target_transformed else 'neg_rmse'
            mae_key = 'neg_mae_orig' if target_transformed else 'neg_mae'
            
            results[name] = {
                'mean_r2': cv_scores.get(r2_key, np.array([np.nan])).mean(),
                'std_r2': cv_scores.get(r2_key, np.array([np.nan])).std(),
                'mean_rmse': -cv_scores.get(rmse_key, np.array([np.nan])).mean(),
                'std_rmse': cv_scores.get(rmse_key, np.array([np.nan])).std(),
                'mean_mae': -cv_scores.get(mae_key, np.array([np.nan])).mean(),
                'std_mae': cv_scores.get(mae_key, np.array([np.nan])).std(),
                'best_params': search.best_params_,
            }
            trained_models[name] = best_estimator
            
        except Exception as e:
            if verbose:
                print(f"FOUT bij training {name}: {e}")
            results[name] = {
                'mean_r2': np.nan, 'std_r2': np.nan,
                'mean_rmse': np.nan, 'std_rmse': np.nan,
                'mean_mae': np.nan, 'std_mae': np.nan,
                'best_params': {}
            }
            trained_models[name] = None
    
    print("--- Model Training & CV Voltooid ---")
    
    # --- Resultaten Samenvatten ---
    if not trained_models:
        if verbose:
            print("Geen modellen beschikbaar.")
        return None, None, None, target_transformed
    
    valid_results = {k: v for k, v in results.items() if not np.isnan(v['mean_rmse'])}
    if not valid_results:
        if verbose:
            print("Geen modellen succesvol geëvalueerd.")
        return None, None, None, target_transformed
    
    summary_df = pd.DataFrame(valid_results).T.sort_values(['mean_rmse', 'std_rmse'], ascending=[True, True])
    best_model_name = summary_df.index[0]
    best_model_pipeline = trained_models.get(best_model_name)
    
    # Toon resultaten
    print("\n" + "="*60)
    print("RESULTATEN MODEL VERGELIJKING")
    print("="*60)
    
    # Print resultaten tabel
    cv_results_table = summary_df[['mean_rmse', 'std_rmse', 'mean_mae', 'std_mae', 'mean_r2', 'std_r2']].copy()
    cv_results_table.columns = ['CV RMSE', 'Std Dev (RMSE)', 'CV MAE', 'Std Dev (MAE)', 'CV R²', 'Std Dev (R²)']
    print(cv_results_table.round(2).to_string())
    
    # --- FOUTENANALYSE SECTIE ---
    if best_model_pipeline:
        print("\n" + "="*50)
        print(" FOUTENANALYSE (Cross-Validation Basis)")
        print("="*50)
        
        # Importeer clone functie
        from sklearn.base import clone
        
        # Maak handmatige CV predictions voor foutenanalyse
        y_pred_cv_list = []
        y_true_orig_cv_list = []
        cv_indices = []
        cv_manual = TimeSeriesSplit(n_splits=n_splits)
        
        for fold, (train_idx, val_idx) in enumerate(cv_manual.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            
            fold_model = clone(best_model_pipeline)
            try:
                # Do NOT use sample_weight during fold training (like original notebook)
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = fold_model.predict(X_val_fold)
                y_pred_cv_list.append(y_pred_fold)
                y_true_orig_fold = y_orig.loc[X.index[val_idx]]
                y_true_orig_cv_list.append(y_true_orig_fold)
                cv_indices.extend(X.index[val_idx])
            except Exception as e_fold:
                print(f"     FOUT tijdens trainen/voorspellen fold {fold+1}: {e_fold}")
                nan_preds = np.full(len(val_idx), np.nan)
                nan_trues = np.full(len(val_idx), np.nan)
                y_pred_cv_list.append(nan_preds)
                y_true_orig_cv_list.append(nan_trues)
                cv_indices.extend(X.index[val_idx])
        
        if not cv_indices:
            print("   FOUT: Geen CV voorspellingen gegenereerd voor foutenanalyse.")
        else:
            y_pred_cv_log_or_orig = pd.Series(np.concatenate(y_pred_cv_list), index=cv_indices).sort_index()
            y_true_orig_cv = pd.Series(np.concatenate(y_true_orig_cv_list), index=cv_indices).sort_index()
            y_pred_cv_orig = np.expm1(y_pred_cv_log_or_orig) if target_transformed else y_pred_cv_log_or_orig
            y_pred_cv_orig[y_pred_cv_orig < 0] = 0
            valid_analysis_idx = y_true_orig_cv.notna() & y_pred_cv_orig.notna()
            y_true_orig_cv = y_true_orig_cv[valid_analysis_idx]
            y_pred_cv_orig = y_pred_cv_orig[valid_analysis_idx]
            
            if y_true_orig_cv.empty:
                print("   FOUT: Geen valide data over voor foutenanalyse na CV.")
            else:
                # --- Gecombineerde Fouten Tabel ---
                absolute_errors = np.abs(y_true_orig_cv - y_pred_cv_orig)
                percentage_errors = np.divide(absolute_errors * 100, y_true_orig_cv, out=np.full_like(absolute_errors, np.nan), where=y_true_orig_cv!=0)
                bins = [400, 800, 1200, 1700, float('inf')]
                labels = ['<800', '800-1200', '1200-1700', '>1700']
                error_df = pd.DataFrame({'Actual': y_true_orig_cv, 'CV_Error': absolute_errors, 'CV_Percentage_Error': percentage_errors}, index=y_true_orig_cv.index)
                error_df['Ticket_Range'] = pd.cut(error_df['Actual'], bins=bins, labels=labels, right=False)
                range_stats = error_df.groupby('Ticket_Range', observed=False).agg(
                     Mean_Abs_Error=('CV_Error', 'mean'), Median_Abs_Error=('CV_Error', 'median'),
                     Mean_Perc_Error=('CV_Percentage_Error', 'mean'), Median_Perc_Error=('CV_Percentage_Error', 'median'),
                     Count=('Actual', 'count')
                )
                
                # Bereken Overall stats uit CV resultaten voor de tabel
                overall_stats = pd.DataFrame({
                    'Mean_Abs_Error': [summary_df.loc[best_model_name, 'mean_mae']],
                    'Median_Abs_Error': [np.nanmedian(absolute_errors)], # Median van de CV fouten
                    'Mean_Perc_Error': [np.nanmean(percentage_errors)],
                    'Median_Perc_Error': [np.nanmedian(percentage_errors)],
                    'Count': [len(y_true_orig_cv)]
                }, index=['OVERALL CV'])
                
                # Combineer en toon
                combined_error_table = pd.concat([range_stats, overall_stats]).round(1)
                print("\nFoutstatistieken per Ticket Range & Overall (Cross-Validation):")
                print(combined_error_table.to_string())
    
    # Probeer ensemble model
    create_ensemble = n_samples >= 40
    if best_model_name and best_model_pipeline and create_ensemble:
        if verbose:
            print("\n--- Probeer ensemble model te creëren ---")
        try:
            ensemble_model, ens_target_transformed = create_ensemble_model(
                X=X,
                y=y,
                preprocessor=preprocessor,
                target_transformed=target_transformed,
                cv=cv,
                verbose=verbose
            )
            
            if isinstance(ensemble_model, dict) and ensemble_model.get('type') == 'ensemble':
                if verbose:
                    print("Evalueer ensemble model met cross-validation...")
                
                # Evalueer het ensemble model
                ensemble_preds = ensemble_cross_predict(
                    X, y, 
                    ensemble_model['base_models'], 
                    ensemble_model['meta_learner'],
                    groups=None,
                    cv=cv
                )
                
                # Bereken metrics voor ensemble
                if target_transformed:
                    # Transform predictions back to original scale
                    y_true_orig = np.expm1(y)
                    y_pred_orig = np.expm1(ensemble_preds)
                    y_pred_orig[y_pred_orig < 0] = 0
                else:
                    y_true_orig = y
                    y_pred_orig = ensemble_preds
                
                from ticket_prediction.models.metrics import rmse_scorer
                from sklearn.metrics import mean_absolute_error, r2_score
                
                ensemble_rmse = rmse_scorer(y_true_orig, y_pred_orig)
                ensemble_mae = mean_absolute_error(y_true_orig, y_pred_orig)
                ensemble_r2 = r2_score(y_true_orig, y_pred_orig)
                
                if verbose:
                    print(f"Ensemble RMSE: {ensemble_rmse:.2f}")
                    print(f"Beste individuele model ({best_model_name}) RMSE: {summary_df.loc[best_model_name, 'mean_rmse']:.2f}")
                
                # Vergelijk met beste individuele model
                if ensemble_rmse < summary_df.loc[best_model_name, 'mean_rmse']:
                    if verbose:
                        print("Ensemble model presteert beter dan beste individuele model!")
                    
                    # Update summary_df met ensemble resultaten
                    summary_df.loc['Ensemble'] = {
                        'mean_rmse': ensemble_rmse,
                        'std_rmse': 0,  # Geen std voor ensemble op deze manier
                        'mean_mae': ensemble_mae,
                        'std_mae': 0,
                        'mean_r2': ensemble_r2,
                        'std_r2': 0,
                        'best_params': {}
                    }
                    
                    # Print updated table
                    cv_results_table = summary_df[['mean_rmse', 'std_rmse', 'mean_mae', 'std_mae', 'mean_r2', 'std_r2']].copy()
                    cv_results_table.columns = ['CV RMSE', 'Std Dev (RMSE)', 'CV MAE', 'Std Dev (MAE)', 'CV R²', 'Std Dev (R²)']
                    print("\n" + "="*60)
                    print("UPDATED RESULTATEN MET ENSEMBLE")
                    print("="*60)
                    print(cv_results_table.round(2).to_string())
                    
                    return cv_results_table, "Ensemble", ensemble_model, target_transformed
                else:
                    if verbose:
                        print(f"Ensemble model ({ensemble_rmse:.2f}) presteert niet beter dan {best_model_name} ({summary_df.loc[best_model_name, 'mean_rmse']:.2f})")
                        print(f"Gebruik {best_model_name} als beste model.")
        except Exception as e:
            if verbose:
                print(f"Fout bij creëren/evalueren van ensemble model: {e}")
    
    return cv_results_table, best_model_name, best_model_pipeline, target_transformed 