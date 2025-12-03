# DRW Cryptocurrency Prediction Challenge 2024 – Top ~12% Private LB

Two-stage solution for the 2024 DRW time-series forecasting competition (crypto price movement prediction).

### 01_feature_selection.ipynb  
- Loads raw 785-dimensional data  
- Trains LightGBM baseline  
- Feature importance via **SHAP values**, **Mutual Information**, and **Permutation Importance**  
- Chronological 80/20 split (no leakage)  
- Final shortlist: **15 robust features** consistently ranked high across all three methods  

### 02_neural_network_final_model.ipynb  
- Deep feed-forward network (TensorFlow/Keras)  
- Optuna hyperparameter search + TFKerasPruningCallback  
- AdamW optimizer, Gaussian noise, BatchNorm, Dropout, L2 regularization  
- Trained only on the 15 selected features  
- Outlier clipping (1%/99% quantiles)  
- Final single-model submission (no ensemble kept simple for interpretability)

Tech stack  
`Python · Pandas · LightGBM · SHAP · Scikit-learn · TensorFlow/Keras · Optuna · Matplotlib/Seaborn`

All notebooks tested and runnable end-to-end on a fresh environment.