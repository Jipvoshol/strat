import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from typing import List

def visualize_linear_model_coefficients(
    model_pipeline: Pipeline, 
    feature_names: List[str],
    verbose: bool = True, 
    model_name: str = "Linear Model", 
    T: int = 7
):
    """Visualiseert de coëfficiënten van een lineair model."""
    if not verbose or not hasattr(model_pipeline.named_steps['reg'], 'coef_'):
        if verbose:
            print(f"INFO: Kan geen coëfficiënten plotten voor {model_name}.")
        return
    
    try:
        regressor = model_pipeline.named_steps['reg']
        coefs = regressor.coef_
        if coefs.ndim > 1:
            coefs = coefs.flatten()  # Handle potential multi-output

        if len(coefs) != len(feature_names):
            print(f"WAARSCHUWING: Aantal coëfficiënten ({len(coefs)}) komt niet overeen met aantal feature names ({len(feature_names)}) voor {model_name}.")
            # Probeer alleen de eerste N coëfficiënten te gebruiken als noodoplossing
            coefs = coefs[:len(feature_names)]

        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(20)  # Top 20

        plt.figure(figsize=(10, 8))
        colors = ['#2ECC71' if c >= 0 else '#E74C3C' for c in coef_df['Coefficient']]  # Groen/Rood
        plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Top {len(coef_df)} Feature Coefficients - {model_name} (T={T})')
        plt.gca().invert_yaxis()  # Belangrijkste bovenaan
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Fout bij visualiseren coëfficiënten voor {model_name}: {e}")

def visualize_tree_model_feature_importance(
    model_pipeline: Pipeline, 
    feature_names: List[str],
    verbose: bool = True, 
    model_name: str = "Tree Model", 
    T: int = 7
):
    """Visualiseert feature importance van een boom-gebaseerd model."""
    if not verbose or not hasattr(model_pipeline.named_steps['reg'], 'feature_importances_'):
        if verbose:
            print(f"INFO: Kan geen feature importance plotten voor {model_name}.")
        return
    
    try:
        regressor = model_pipeline.named_steps['reg']
        importances = regressor.feature_importances_

        if len(importances) != len(feature_names):
            print(f"WAARSCHUWING: Aantal importances ({len(importances)}) komt niet overeen met aantal feature names ({len(feature_names)}) voor {model_name}.")
            # Probeer alleen de eerste N importances te gebruiken als noodoplossing
            importances = importances[:len(feature_names)]

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(20)  # Top 20

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3498DB')  # Blauw
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {len(importance_df)} Feature Importances - {model_name} (T={T})')
        plt.gca().invert_yaxis()  # Belangrijkste bovenaan
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Fout bij visualiseren feature importance voor {model_name}: {e}") 