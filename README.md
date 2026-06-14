# Urban Traffic Demand Forecasting

Spatio-temporal ML pipeline for predicting urban traffic demand 
using geohash-based spatial features and time-series lag modeling.

## Results
- **R² Score**: 0.9197 (5-Fold CV)
- **Approach**: Dual-Model Blend (Lag-aware + Global LightGBM) 
  with Pseudo-Label Retraining

## Architecture
1. OOP Preprocessing Pipeline (custom sklearn transformers)
2. MICE Imputation with ExtraTrees
3. Spatial Feature Engineering (geohash decoding + KMeans clustering)
4. Temporal Feature Engineering (cyclical encoding, lag features)
5. Dual LightGBM Ensemble (80% Lag + 20% Global)
6. Pseudo-Label Retraining on combined train+test

## Status
🚧 Active development. Future work: 
- Transformer-based temporal encoder
- Graph Neural Networks for geohash adjacency
- SHAP-based interpretability layer

## Tech Stack
Python, LightGBM, scikit-learn, pygeohash, pandas, numpy