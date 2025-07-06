import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List

class RmseOptimizedPreprocessor:
    """Class voor het creëren van geoptimaliseerde preprocessing pipelines voor RMSE reductie"""
    
    @staticmethod
    def create_pipeline(numerical_features: List[str], 
                       categorical_features: List[str], 
                       use_robust: bool = True, 
                       use_power_transform: bool = True):
        """
        Creëert een preprocessing pipeline geoptimaliseerd voor RMSE reductie.
        
        Parameters:
            numerical_features: Lijst met numerieke features
            categorical_features: Lijst met categorische features
            use_robust: Of RobustScaler gebruikt moet worden
            use_power_transform: Of PowerTransformer gebruikt moet worden
            
        Returns:
            ColumnTransformer: De geoptimaliseerde preprocessing pipeline
        """
        num_steps = []
        # KNNImputer is beter voor numerieke data
        num_steps.append(('imputer', KNNImputer(n_neighbors=5)))
        
        if use_power_transform:
            # Power transform voor betere normale verdeling
            num_steps.append(('power', PowerTransformer(method='yeo-johnson', standardize=False)))
        
        if use_robust:
            # RobustScaler is minder gevoelig voor outliers
            num_steps.append(('scaler', RobustScaler()))
        else:
            num_steps.append(('scaler', StandardScaler()))
            
        # Categorische features transformatie
        cat_steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
        
        transformers = [
            ('num', Pipeline(steps=num_steps), numerical_features),
            ('cat', Pipeline(steps=cat_steps), categorical_features)
        ]
        
        return ColumnTransformer(transformers=transformers, remainder='drop') 