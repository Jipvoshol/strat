# Ticket Prediction Package

Dit pakket is een gerefactorde versie van het `strategic_big.ipynb` notebook, opgesplitst in modulaire Python bestanden met een logische structuur.

## Structuur

```
ticket_prediction/
├── __init__.py                      # Package initialisatie
├── config/
│   ├── __init__.py
│   └── constants.py                 # Globale constanten (RANDOM_STATE, best_models)
├── data/
│   ├── __init__.py
│   ├── loader.py                    # Data loading functionaliteit
│   └── preprocessing.py             # Data cleaning en voorbewerking
├── features/
│   ├── __init__.py
│   ├── engineering.py              # Hoofdfunctie voor feature engineering
│   ├── time_features.py            # Tijd-gebaseerde features
│   ├── sales_features.py           # Verkoop-gebaseerde features
│   ├── artist_features.py          # Artiest-gebaseerde features
│   ├── demographic_features.py     # Demografische features
│   ├── ticket_features.py          # Ticket-gebaseerde features
│   └── weather_features.py         # Weer en capacity features
├── models/
│   ├── __init__.py
│   ├── training.py                 # Model training en vergelijking
│   ├── prediction.py               # Voorspelling functionaliteit
│   ├── preprocessors.py            # Data preprocessing pipelines
│   ├── metrics.py                  # Custom metrics en scorers
│   ├── selection.py                # Feature selectie methodes
│   └── ensemble.py                 # Ensemble model implementatie
├── utils/
│   ├── __init__.py
│   ├── text_processing.py          # Tekst normalisatie
│   ├── geo_utils.py                # Geografische utilities
│   ├── outlier_detection.py        # Outlier detectie
│   └── model_persistence.py        # Model opslaan/laden
├── visualization/
│   ├── __init__.py
│   ├── model_interpretation.py     # Model interpretatie visualisaties
│   └── prediction_plots.py         # Voorspelling visualisaties
├── main.py                         # Hoofd script
└── requirements.txt                # Dependencies
```

## Gebruik

### Basis gebruik

```python
from ticket_prediction import main, predict_tickets

# Train modellen
best_models = main(
    tickets_path='tickets_processed.csv',
    line_up_path='line_up_processed_new.csv',
    artists_path='artist_insights.csv',
    T_values=[7, 14, 30],  # Forecast horizons
    verbose=True
)

# Maak voorspellingen
predictions = predict_tickets(
    df_events=event_data,
    T=14,
    known_cities=cities,
    line_up_df=lineup,
    artists_df=artists,
    verbose=True
)
```

### Direct runnen

```bash
cd ticket_prediction
python3 main.py
```

Dit zal:
1. Data laden
2. Feature engineering uitvoeren
3. Modellen trainen voor T=14
4. Het beste model opslaan
5. Voorspellingen maken

## Belangrijke functies

### Data Loading
- `load_data()`: Laadt de drie hoofdbestanden (tickets, lineup, artists)

### Feature Engineering
- `engineer_features()`: Hoofdfunctie die alle feature engineering orchestreert
- Aparte modules voor verschillende feature types (tijd, verkoop, artiest, etc.)

### Model Training
- `run_model_comparison()`: Vergelijkt verschillende modellen met cross-validation
- Ondersteunt: Lasso, Ridge, ElasticNet, RandomForest, GradientBoost, XGBoost, CatBoost

### Predictions
- `predict_tickets()`: Maakt voorspellingen voor nieuwe events
- Handelt automatisch log-transformaties af

## Dependencies

Zie `requirements.txt` voor alle benodigde packages:
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- matplotlib, plotly
- geonamescache
- en meer...

## Verschillen met het notebook

Deze gerefactorde versie:
- Heeft een modulaire structuur voor betere onderhoudbaarheid
- Bevat robuuste error handling
- Is volledig gedocumenteerd met type hints
- Kan eenvoudig worden uitgebreid
- Behoudt EXACT dezelfde functionaliteit als het originele notebook

## Testing

Run de test script om te verifiëren dat alles werkt:

```bash
python3 test_refactoring.py
```

## Model Persistence

Modellen worden automatisch opgeslagen als:
```
ticket_prediction_model_T{dagen}_CV_{ModelNaam}.pkl
```

En kunnen worden geladen met:
```python
from ticket_prediction.utils.model_persistence import load_model
load_model('ticket_prediction_model_T14_CV_XGBRegressor.pkl')
``` 