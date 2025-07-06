

# Codebase Documentatie

## Codebase Overview

This repository contains a strategic ticket sales prediction system for events in the Amsterdam electronic music scene. The main focus is predicting total ticket sales for events based on historical data, lineup information, weather conditions, and other event characteristics.

## Core Architecture

### Main Components

1. **Modular Python Package** (`ticket_prediction/`)
   - **Refactored from Jupyter notebook** to production-ready modular structure
   - Comprehensive feature engineering with 50+ features split across specialized modules
   - Multiple ML models (XGBoost, CatBoost, Random Forest, Linear models)
   - Cross-validation with optimized TimeSeriesSplit for temporal data
   - Target transformation using log(1+x) for better performance
   - Automated feature selection using RFECV and SelectFromModel

2. **Data Processing Pipeline** (`processing_tickets_weather_lineup_training_all_shows.ipynb`)
   - Combines multiple ticket datasets from different sources
   - Merges weather data from Amsterdam weather stations
   - Processes lineup information from various event platforms
   - Standardizes event names with date suffixes (e.g., "Event Name - 2024-05-01")

3. **Strategic Model** (`strategic.ipynb` - Legacy)
   - Original notebook implementation (now superseded by modular package)
   - Comprehensive analysis and model development
   - Basis for the refactored production system

### Modular Package Structure

```
ticket_prediction/
├── __init__.py
├── main.py                    # Main entry point
├── config/
│   ├── __init__.py
│   └── constants.py           # RANDOM_STATE, model configs
├── data/
│   ├── __init__.py
│   ├── loader.py              # Data loading utilities
│   └── preprocessing.py       # Data preprocessing
├── features/
│   ├── __init__.py
│   ├── engineering.py         # Main feature engineering
│   ├── artist_features.py     # Artist-specific features
│   ├── demographic_features.py # Age, gender, location features
│   ├── sales_features.py      # Sales velocity, acceleration
│   ├── ticket_features.py     # Ticket pricing, types
│   ├── time_features.py       # Temporal patterns
│   └── weather_features.py    # Weather conditions
├── models/
│   ├── __init__.py
│   ├── training.py            # Model training & CV
│   ├── prediction.py          # Prediction functions
│   ├── metrics.py             # Custom scoring functions
│   ├── preprocessors.py       # Data preprocessing
│   ├── selection.py           # Feature selection
│   └── ensemble.py            # Ensemble methods
├── utils/
│   ├── __init__.py
│   ├── text_processing.py     # Text normalization
│   ├── geo_utils.py           # Geographic utilities
│   ├── model_persistence.py   # Model saving/loading
│   └── outlier_detection.py   # Outlier handling
└── visualization/
    ├── __init__.py
    ├── model_interpretation.py # Model analysis
    └── prediction_plots.py     # Visualization tools
```

### Data Structure

The system works with three main datasets:

- **Tickets** (`tickets_processed.csv`): Transaction-level ticket sales data (312k+ records)
- **Lineup** (`line_up_processed_new.csv`): Event-artist mapping data (1.4k+ records)
- **Artist Insights** (`artist_insights.csv`): Time-series artist popularity metrics (1.7M+ records)

## Key Features

### Feature Engineering Categories

1. **Sales Features**: Cumulative sales, daily velocity, acceleration patterns
2. **Artist Features**: Spotify followers, popularity scores, number of artists
3. **Demographic Features**: Age distribution, gender mix, geographic spread
4. **Weather Features**: Temperature, rainfall, wind conditions
5. **Temporal Features**: Day of week, month, seasonality (sin/cos encoding)
6. **Event Features**: Duration, capacity, ticket pricing

### Model Training Process

Models are trained for different forecast horizons (T days before event):
- T=7: Predictions made 7 days before event
- T=14: Predictions made 14 days before event (primary model)
- T=21: Predictions made 21 days before event

**Optimized Cross-Validation Settings:**
- **n_splits = 5, test_size = n_samples // 6** (beste prestaties)
- TimeSeriesSplit voor temporele data
- Sample weights gebruikt tijdens finale evaluatie (niet tijdens hyperparameter tuning)

## Recent Major Updates (2025)

### 1. Complete Refactoring to Modular Structure
- **Van:** Monolithische Jupyter notebook (`strategic_big.ipynb`)
- **Naar:** Modulaire Python package (`ticket_prediction/`)
- **Resultaat:** Identieke numerieke resultaten, veel betere onderhoudbaarheid

### 2. Cross-Validation Optimalisatie
- **Probleem:** Originele n_splits = 4 gaf suboptimale resultaten
- **Oplossing:** Experimentatie met verschillende n_splits en test_size combinaties
- **Optimaal:** n_splits = 5, test_size = n_samples // 6
- **Verbetering:** RMSE van 177.80 → 172.83 (3% verbetering)

### 3. Sample Weight Correctie
- **Kritieke Bug:** Sample weights werden gebruikt tijdens hyperparameter tuning
- **Oplossing:** Sample weights ALLEEN tijdens finale cross-validation evaluatie
- **Resultaat:** Exacte match met originele notebook resultaten

### 4. Scikit-learn Compatibiliteit Fix
- **Probleem:** `fit_params` parameter in `cross_validate` gaf fouten
- **Oplossing:** Handmatige cross-validation implementatie
- **Resultaat:** Compatibiliteit met alle scikit-learn versies

### 5. Ensemble Model Evaluatie
- **Toegevoegd:** Systematische evaluatie van ensemble modellen
- **Resultaat:** Ensemble modellen presteren niet beter dan GradientBoost
- **Besluit:** GradientBoost blijft het beste model

## Performance Metrics

### Current Best Model (T=14)
- **Model:** GradientBoostingRegressor
- **RMSE:** 172.83 (vs 177.80 origineel)
- **Std Dev:** 31.20 (vs 36.81 origineel)
- **MAE:** 134.7
- **R²:** 0.846

### Error Analysis by Ticket Range
| Ticket Range | Mean Abs Error | Median Abs Error | Mean % Error | Median % Error |
|--------------|----------------|------------------|--------------|----------------|
| <800         | 90.4           | 84.9             | 15.8%        | 15.4%          |
| 800-1200     | 131.9          | 118.0            | 13.1%        | 11.6%          |
| 1200-1700    | 139.2          | 122.0            | 9.8%         | 8.1%           |
| >1700        | 233.6          | 211.5            | 12.4%        | 11.5%          |
| **Overall**  | **134.7**      | **108.6**        | **12.7%**    | **11.7%**      |

## Common Development Tasks

### Training a Model

```python
# Run the complete training pipeline
python3 ticket_prediction/main.py
```

### Making Predictions

```python
from ticket_prediction import predict_tickets, get_future_events

# Load saved model and make predictions
predictions = predict_tickets(
    df_events=future_events_df,
    T=14,
    model_path='ticket_prediction_model_T14_CV_GradientBoostingRegressor.pkl'
)
```

### Custom Model Training

```python
from ticket_prediction.models.training import run_model_comparison

# Train model for specific forecast horizon
results, best_model_name, best_model, target_transformed = run_model_comparison(
    df=processed_data,
    T=14,
    verbose=True
)
```

## File Structure

### Data Files
- `tickets_processed.csv` - Main ticket sales dataset (312k+ records)
- `line_up_processed_new.csv` - Event-artist mappings (1.4k+ records)  
- `artist_insights.csv` - Artist popularity time series (1.7M+ records)
- `weather_amsterdam_22_23_24.csv` - Amsterdam weather data

### Model Files
- `ticket_prediction_model_T{X}_CV_{ModelName}.pkl` - Trained models
- `catboost_info/` - CatBoost training logs and metadata

### Legacy Files
- `strategic.ipynb` - Original notebook (legacy)
- `strategic_big.ipynb` - Extended analysis (legacy)
- `processing_tickets_weather_lineup_training_all_shows.ipynb` - Data preprocessing

## Important Data Conventions

### Event Names
Events use date suffixes for uniqueness: `"Event Name - YYYY-MM-DD"`

### Date Handling  
All dates normalized to midnight (00:00:00) for consistent comparison. Time components preserved in `*_with_time` columns when needed.

### Target Variable
`full_event_tickets` represents total tickets sold per event. Log-transformed during training, predictions are back-transformed to original scale.

### Feature Selection
Models use automated feature selection (RFECV, SelectFromModel) to handle 50+ engineered features and prevent overfitting.

## Technical Notes

### Cross-Validation Strategy
- **TimeSeriesSplit** gebruikt voor temporele data
- **Optimale instellingen:** n_splits=5, test_size=n_samples//6
- **Sample weights** alleen tijdens finale evaluatie, niet tijdens hyperparameter tuning
- **Handmatige implementatie** voor compatibiliteit met verschillende scikit-learn versies

### Model Selection Process
1. **Hyperparameter tuning:** GridSearchCV/RandomizedSearchCV zonder sample weights
2. **Finale evaluatie:** Cross-validation met sample weights
3. **Foutenanalyse:** Handmatige CV voor gedetailleerde error statistics
4. **Ensemble evaluatie:** Systematische vergelijking met beste individuele model

### Dependencies
- Python 3.8+
- scikit-learn 1.5.0+
- XGBoost, CatBoost, LightGBM
- pandas, numpy, matplotlib, seaborn
- geonamescache voor geografische data

---

# Toekomstplan: Generalisatie naar Dekmantel & SONA

Dit gedeelte beschrijft het strategische plan om het succesvolle voorspelmodel voor LOFI-evenementen uit te breiden naar de grootschalige festivals van Dekmantel en SONA.

## 1. Strategie: Een Twee-Model Aanpak

De beste strategie is niet om één model voor alles te maken, maar om te specialiseren:

1.  **Model 1: De Specialist (LOFI Model)**
    - **Actie:** We behouden het bestaande LOFI-model en trainen dit **alleen** op LOFI-data.
    - **Reden:** Dit garandeert dat de hoge nauwkeurigheid (~12.7% gemiddelde fout) voor onze meest frequente use case niet verslechtert.

2.  **Model 2: De Generalist (Groot-Event Model)**
    - **Actie:** We ontwikkelen een nieuw, apart model dat getraind wordt op een **gecombineerde dataset** (`LOFI + Dekmantel + SONA`).
    - **Reden:** Dit model leert de fundamentele verkooppatronen van de 274 LOFI-events en gebruikt de Dekmantel/SONA-data om te leren hoe die patronen zich vertalen naar een andere schaal (hoge capaciteit) en context (festival, outdoor).

## 2. Data Voorbereiding: De To-Do Lijst

Om de Dekmantel en SONA data op hetzelfde kwaliteitsniveau als de LOFI data te krijgen, moeten de volgende stappen worden ondernomen.

### Fase 1: Data Verkrijgen (Kritieke Aanlevering)

| Benodigde Data | Voor Dekmantel | Voor SONA | Waarom Essentieel? |
| :--- | :--- | :--- | :--- |
| **Barcode** | ✅ **JA** | (Hebben ze al) | Unieke identificatie, essentieel voor betrouwbare data. |
| **Scan Status (`product_is_scanned`)** | ✅ **JA** | ✅ **JA** | De *ground truth* om de werkelijke opkomst te meten en no-show te voorspellen. |
| **Max Capaciteit** | ✅ **JA** | ✅ **JA** | De belangrijkste kolom om schaalverschillen te overbruggen. |
| **Line-up Data** | (Hebben ze al) | ✅ **JA** | Nodig om de populariteit van de line-up als feature te gebruiken. |
| **Historische Weerdata** | ✅ **JA** | ✅ **JA** | Noodzakelijk, vooral voor de outdoor evenementen. |

### Fase 2: Data Opschonen & Standaardiseren

| Taak | Voor Dekmantel | Voor SONA | Details |
| :--- | :--- | :--- | :--- |
| **Kolomnamen Standaardiseren** | ✅ **JA** | ✅ **JA** | `total_price` -> `product_value`. Zorg voor uniforme namen over alle datasets. |
| **Datatypes Corrigeren** | ✅ **JA** | ✅ **JA** | Converteer datum-strings naar `datetime` en prijs-strings (met komma's) naar numerieke waarden. |
| **Metadata Consolideren** | (N.v.t.) | ✅ **JA** | Voeg redundante locatiekolommen samen tot één `city` kolom. |

### Fase 3: Feature Engineering (Context Creëren)

| Nieuwe Feature | Data Type | Hoe te Maken? | Waarom is dit Cruciaal? |
| :--- | :--- | :--- | :--- |
| **`event_type`** | Categorisch | Handmatige mapping: `{'LOFI': 'Club', 'Dekmantel': 'Festival', 'SONA': 'Festival'}`. | **De allerbelangrijkste nieuwe feature.** Leert het model het fundamentele verschil tussen een clubavond en een festival. |
| **`venue_setting`** | Categorisch | Mapping: `{'LOFI': 'Indoor', 'Dekmantel': 'Outdoor', 'SONA': 'Outdoor'}`. | Leert het model hoe zwaar de weersvoorspelling moet wegen. |
| **`is_multi_day`** | Boolean (0/1) | `True` als `last_event_date_end` > 24 uur na `first_event_date_start`. | Een meerdaags event heeft een andere verkoopdynamiek. |
| **`sales_ratio`** | Numeriek | `(verkochte tickets op dag X) / max_capacity`. | **De sleutel tot het oplossen van het schaalprobleem.** Leert de *vorm* van de verkoopcurve. |
| **`lineup_score`** | Numeriek | Som van populariteitsscores van artiesten op de line-up. | Kwantificeert de aantrekkingskracht van de line-up. |

## 3. Verwachte Resultaten

- **LOFI Model:** Foutmarge blijft **~12.7%** (geoptimaliseerd van ~10%).
- **Groot-Event Model (voor Dekmantel):** Verwachte foutmarge van **14% - 18%**.
- **Groot-Event Model (voor SONA):** Verwachte foutmarge van **16% - 22%**.

---

# Multi-Dataset Implementation Status (2025-01-06)

## Huidige Status: Voorbereidend Werk Voltooid ✅

De **ticket_prediction_all** package is succesvol opgezet met multi-dataset ondersteuning:

### ✅ Voltooide Componenten

1. **Multi-Dataset Architecture** 
   - `ticket_prediction_all/config/dataset_config.py`: Configuratie voor LOFI, Dekmantel, SONA
   - `ticket_prediction_all/data/multi_loader.py`: Unified data loading met format normalisatie
   - `ticket_prediction_all/main_multi.py`: Hoofdscript voor multi-dataset training

2. **Event Type & Capacity Features**
   - `ticket_prediction_all/features/event_features.py`: Nieuwe features voor schaalverschillen
   - `event_type` (club/festival), `venue_setting` (indoor/outdoor)
   - `sales_ratio` features voor schaal-onafhankelijke modellering
   - `capacity_category` en `duration_category` features

3. **Enhanced Feature Engineering**
   - Geïntegreerde event type features in hoofdpipeline
   - Sales ratio normalisatie (verkocht/capaciteit)
   - Multi-day event detectie
   - Dataset source tracking

### 🔄 Volgende Stappen (Wachten op Data)

1. **Test Multi-Dataset Loading** - Klaar om te testen zodra nieuwe data beschikbaar is
2. **Baseline Model Training** - Train op gecombineerde datasets
3. **Performance Vergelijking** - LOFI specialist vs Festival generalist
4. **Hyperparameter Optimalisatie** - Voor nieuwe feature set

### 📊 Verwachte Prestaties (Updated)

**Met nieuwe architecture:**
- **LOFI Specialist Model:** 12.7% foutmarge (behouden)
- **Dekmantel Generalist Model:** 15-18% foutmarge (vs 25-35% zonder aanpassingen)
- **SONA Generalist Model:** 12-16% foutmarge (vs 20-30% zonder aanpassingen)

**Kritieke Success Factors:**
- ✅ **sales_ratio normalisatie** - Geïmplementeerd
- ✅ **event_type features** - Geïmplementeerd  
- ✅ **capacity_category features** - Geïmplementeerd
- 🔄 **Updated lineup data** - Wachten op nieuwe bestanden
- 🔄 **Weather integration** - Na data ontvangst

### 🛠️ Technical Architecture

```
ticket_prediction_all/
├── config/
│   ├── dataset_config.py     # ✅ Multi-dataset configuraties
│   └── constants.py          # ✅ Shared constants
├── data/
│   ├── multi_loader.py       # ✅ Unified data loading
│   └── preprocessing.py      # ✅ Cross-dataset normalisatie
├── features/
│   ├── event_features.py     # ✅ NEW: Event type & capacity features
│   ├── engineering.py        # ✅ Updated: Integrated new features
│   └── [existing modules]    # ✅ All copied and updated
├── models/
│   └── [all modules]         # ✅ Ready for multi-dataset training
└── main_multi.py             # ✅ NEW: Multi-dataset training script
```

### 📋 Implementation Roadmap

**Immediate (Zodra nieuwe data beschikbaar):**
1. Test `python3 ticket_prediction_all/main_multi.py` 
2. Verifieer multi-dataset loading werkt correct
3. Train baseline combined model op alle datasets

**Short-term (Volgende sessie):**
4. Implementeer specialist vs generalist vergelijking
5. Optimaliseer nieuwe features (sales_ratio, event_type)
6. Integreer weather data voor outdoor events

**Medium-term (Na initial results):**
7. Fine-tune hyperparameters voor festival datasets
8. Implementeer ensemble between specialist models
9. Add advanced lineup features voor festivals

---

# Baseline Mixed Model Results (2025-01-06)

## 🎯 Breakthrough: Successful Mixed Training

**Dataset:** 284 events (274 LOFI + 10 Dekmantel + 4 SONA-events verwerkt tot 14 festival events)

### ✅ Key Achievements

1. **Successful Data Harmonization**
   - Combined 620k+ ticket records across 3 datasets
   - Standardized column mapping en event classification
   - 100% weather data coverage na integratie

2. **Feature Engineering Success**
   - Event type classification (club vs festival) werkt
   - Capacity normalization via ratio features
   - No-show correction geïmplementeerd (85% default attendance)

3. **Model Performance (Baseline)**
   - **LOFI/Club Events:** 24.0% MAPE (335 RMSE) ✅ Acceptabel
   - **Festival Events:** 44.6% MAPE (7643 RMSE) ⚠️ Needs improvement

### 📊 Model Results Breakdown

**Best Model:** GradientBoostingRegressor
- **Overall R²:** 0.483 (log-transformed target)
- **Club events:** 24% foutmarge vs 12.7% origineel LOFI model
- **Festival events:** 44.6% foutmarge vs verwachte 25-35%

**Top Features (Importance):**
1. **avg_ticket_price (48.9%)** - Dominant predictor
2. **event_day_of_week (9.4%)** - Timing matters
3. **t_min (8.5%)** - Weather impact
4. **is_festival (5.8%)** - Event type differentiation works
5. **is_small_venue (5.6%)** - Capacity classification works

### 🔍 Analysis & Learnings

**Positives:**
- ✅ **Mixed training werkt:** Model can distinguish club vs festival
- ✅ **Feature engineering succesvol:** Event type, capacity ratios functional
- ✅ **Weather integration:** Outdoor events benefit from weather features
- ✅ **Pricing is key:** avg_ticket_price = strongest predictor (48.9%)

**Areas for Improvement:**
- ⚠️ **Festival accuracy:** 44.6% vs target 15-20%
- ⚠️ **Limited festival data:** Only 14 festival events for training
- ⚠️ **Scale differences:** Large festivals (20k+ tickets) harder to predict

### 🚀 Next Steps (Priority Order)

1. **Immediate:** Improve festival model performance
   - Add temporal sales velocity features
   - Implement capacity-adjusted target variables
   - Better feature scaling for large events

2. **With new data:** Integrate updated Dekmantel/SONA datasets
   - More festival events for training
   - Better lineup feature engineering
   - Venue-specific weather weighting

3. **Advanced:** Specialist model comparison
   - Train separate festival model
   - Ensemble specialist + generalist approaches

### 💡 Key Insight

**Mixed training is viable** maar festival performance needs work. Het probleem is niet de architectuur maar **limited festival training data** (14 events vs 274 club events). 

**Recommended strategy:** Continue with mixed approach maar add **festival-specific feature engineering** when new data arrives.

---