import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from ticket_prediction.models.metrics import rmse_scorer

def visualize_predictions(
    y_true: pd.Series, 
    y_pred: pd.Series,
    model_name: str, 
    T: int, 
    context: str = "Cross-Validation"
):
    """
    Visualiseert voorspellingen versus werkelijke waarden en residuals.
    
    Parameters:
        y_true: Werkelijke waarden
        y_pred: Voorspelde waarden
        model_name: Naam van het model
        T: Forecast horizon
        context: Context voor de visualisatie
    """
    r2 = r2_score(y_true, y_pred)
    rmse = rmse_scorer(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Plot Voorspeld vs Werkelijk
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label=f'{context} Predictions')
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    min_val = min(0, y_true.min(), y_pred.min()) * 0.95 
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfecte Voorspelling')
    plt.xlabel('Werkelijke Waarden')
    plt.ylabel('Voorspelde Waarden')
    plt.title(f'Voorspeld vs. Werkelijk ({context}) - {model_name} (T={T})')
    plt.text(0.05, 0.95, f'R²={r2:.3f}\nRMSE={rmse:.1f}\nMAE={mae:.1f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=min_val)
    plt.ylim(bottom=min_val)
    plt.tight_layout()
    plt.show()

    # Plot Residuals vs Voorspeld
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Voorspelde Waarden')
    plt.ylabel('Residuals (Werkelijk - Voorspeld)')
    plt.title(f'Residual Plot ({context}) - {model_name} (T={T})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_error_by_ticket_range(
    y_true: pd.Series, 
    y_pred: pd.Series,
    model_name: str,
    T: int
):
    """
    Plot fouten per ticket range.
    
    Parameters:
        y_true: Werkelijke waarden
        y_pred: Voorspelde waarden
        model_name: Naam van het model
        T: Forecast horizon
    """
    # Bereken fouten
    absolute_errors = np.abs(y_true - y_pred)
    percentage_errors = np.divide(
        absolute_errors * 100, 
        y_true, 
        out=np.full_like(absolute_errors, np.nan), 
        where=y_true!=0
    )
    
    # Maak bins
    bins = [400, 800, 1200, 1700, float('inf')]
    labels = ['<800', '800-1200', '1200-1700', '>1700']
    
    # Categoriseer
    error_df = pd.DataFrame({
        'Actual': y_true, 
        'Error': absolute_errors, 
        'Percentage_Error': percentage_errors
    })
    error_df['Ticket_Range'] = pd.cut(error_df['Actual'], bins=bins, labels=labels, right=False)
    
    # Plot absolute error per range
    plt.figure(figsize=(10, 6))
    error_df.boxplot(column='Error', by='Ticket_Range', ax=plt.gca())
    plt.title(f'Absolute Error per Ticket Range - {model_name} (T={T})')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Ticket Range')
    plt.ylabel('Absolute Error')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # Plot percentage error per range
    plt.figure(figsize=(10, 6))
    error_df.boxplot(column='Percentage_Error', by='Ticket_Range', ax=plt.gca())
    plt.title(f'Percentage Error per Ticket Range - {model_name} (T={T})')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Ticket Range')
    plt.ylabel('Percentage Error (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_df: pd.DataFrame, metric: str = 'RMSE'):
    """
    Plot vergelijking tussen modellen.
    
    Parameters:
        results_df: DataFrame met model resultaten
        metric: Metric om te plotten ('RMSE', 'MAE', 'R2')
    """
    plt.figure(figsize=(10, 6))
    
    # Bepaal kolommen voor metric
    if metric == 'RMSE':
        mean_col = 'mean_rmse'
        std_col = 'std_rmse'
        ylabel = 'RMSE'
    elif metric == 'MAE':
        mean_col = 'mean_mae'
        std_col = 'std_mae'
        ylabel = 'MAE'
    elif metric == 'R2':
        mean_col = 'mean_r2'
        std_col = 'std_r2'
        ylabel = 'R²'
    else:
        raise ValueError(f"Onbekende metric: {metric}")
    
    # Plot bar chart met error bars
    models = results_df.index
    means = results_df[mean_col]
    stds = results_df[std_col]
    
    bars = plt.bar(models, means, yerr=stds, capsize=10, 
                    color='#3498DB', edgecolor='black', linewidth=1)
    
    # Voeg waarden toe boven bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(f'Model Comparison - {metric}')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show() 