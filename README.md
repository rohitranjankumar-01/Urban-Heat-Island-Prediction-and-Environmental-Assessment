# Urban Heat Island Prediction & Environmental Assessment

A machine learning project to predict Land Surface Temperature (LST) anomalies across 50 global cities (2015–2025) using satellite-derived land cover metrics, vegetation indices, and climate zones.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitranjankumar-01/Urban-Heat-Island-Prediction-and-Environmental-Assessment/blob/main/Urban_Heat_Island_Prediction_and_Environmental_Assessment.ipynb)
[![View on nbviewer](https://img.shields.io/badge/render-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/rohitranjankumar-01/Urban-Heat-Island-Prediction-and-Environmental-Assessment/blob/main/Urban_Heat_Island_Prediction_and_Environmental_Assessment.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

Urban Heat Islands (UHI) occur when cities experience higher temperatures than surrounding rural areas due to dense concrete and asphalt surfaces, reduced vegetation, and human activity.

This notebook builds and evaluates regression models to predict `lst_anomaly_c` (°C above city baseline) from satellite and socio-environmental data.

### Quick Links
- **Jupyter Notebook**: [Urban_Heat_Island_Prediction_and_Environmental_Assessment.ipynb](Urban_Heat_Island_Prediction_and_Environmental_Assessment.ipynb)
- **Dataset**: [Global Urban Heat Intensity Dataset (2015–2025) on Kaggle](https://www.kaggle.com/datasets/alitaqishah/global-urban-heat-intensity-dataset-20152025)

---

## Dataset

The dataset (`global_urban_heat_island_2015_2025.csv`) contains 550 records across 50 cities over 11 years.

- **Target**: `lst_anomaly_c` — Land Surface Temperature anomaly in °C
- **Features**:
  - `ndvi_mean` — Mean vegetation index (0 to 1)
  - `impervious_pct` — Concrete/asphalt cover (%)
  - `tree_canopy_pct` — Tree canopy cover (%)
  - `pop_density_km2` — Population density (people/km²)
  - `climate_zone` — Köppen-Geiger climate classification (15 zones)
  - `cooling_deficit_ratio` — Engineered ratio: `impervious_pct / (tree_canopy_pct + 1e-4)`

---

## Model Performance

Models were trained on an 80/20 train-test split (440 train, 110 test) using a scikit-learn `ColumnTransformer` (StandardScaler for numerical features, OneHotEncoder for climate zones) and tuned via 5-fold cross-validation.

| Model | Test R² | MAE (°C) | RMSE (°C) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Support Vector Regressor (SVR)** | **0.8463** | **0.1420** | **0.1806** | Best overall (RBF kernel, C=50, epsilon=0.1) |
| **Random Forest Regressor** | 0.8324 | 0.1468 | 0.1886 | Bagging ensemble (100 trees, max depth 10) |
| **Gradient Boosting Regressor** | 0.8169 | 0.1539 | 0.1972 | Boosting ensemble (100 trees, lr=0.05) |
| **Decision Tree (Tuned)** | 0.8150 | 0.1579 | 0.1981 | Pre-pruned (max depth 5) to prevent overfitting |
| **Linear Regression** | 0.5809 | 0.2359 | 0.2982 | Baseline OLS |

---

## Key Takeaways

1. **Main Driver**: Built-up concrete cover (`impervious_pct`) is the strongest predictor, accounting for ~30.6% of feature importance, followed by population density (~21.1%).
2. **Cooling Deficit**: The engineered ratio of concrete to tree canopy (`cooling_deficit_ratio`) contributed ~10% importance, showing that the balance between hard surfaces and canopy cover provides valuable signal beyond individual features.
3. **Best Model**: SVR with an RBF kernel performed best (R² = 0.8463, MAE = 0.142 °C), fitting smooth spatial temperature variations better than tree-based partitions.
4. **Pruning Decision Trees**: An unconstrained Decision Tree overfit heavily (train $R^2 = 1.00$, test $R^2 = 0.748$). Hyperparameter tuning closed the generalization gap and improved test $R^2$ to 0.815.

---

## License

This project is licensed under the [MIT License](LICENSE).
