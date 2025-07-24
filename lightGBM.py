import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from scipy.stats import pearsonr
import optuna
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess (same as your NN code)
train_df = pd.read_parquet("train.parquet")

float_cols = [col for col in train_df.columns if col.startswith('X') or col == 'label']
dtypes = {col: 'float32' for col in float_cols}
if 'timestamp' in train_df.columns:
    dtypes['timestamp'] = 'int64'
if 'asset_id' in train_df.columns:
    dtypes['asset_id'] = 'int32'
train_df = train_df.astype(dtypes)
test_df = pd.read_parquet("test.parquet")
test_dtypes = {col: 'float32' for col in test_df.columns if col.startswith('X')}
if 'timestamp' in test_df.columns:
    test_dtypes['timestamp'] = 'int64'
if 'asset_id' in test_df.columns:
    test_dtypes['asset_id'] = 'int32'
if 'id' in test_df.columns:
    test_dtypes['id'] = 'int64'
test_df = test_df.astype(test_dtypes)

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(train_df.median(numeric_only=True), inplace=True)

inf_cols = [f"X{i}" for i in range(697, 718) if f"X{i}" in train_df.columns]
train_df.drop(columns=inf_cols, errors='ignore', inplace=True)
test_df.drop(columns=inf_cols, errors='ignore', inplace=True)

X = train_df.drop(columns=["label"])
y = train_df["label"]

# Subsample for tuning
subsample_idx = np.random.choice(X.index, size=int(0.5 * len(X)), replace=False)
X_sub = X.loc[subsample_idx]
y_sub = y.loc[subsample_idx]

X_train, X_valid, y_train, y_valid = train_test_split(X_sub, y_sub, test_size=0.1, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
selector = VarianceThreshold(threshold=1e-4)
X_train_reduced = selector.fit_transform(X_train_scaled)

X_valid_scaled = scaler.transform(X_valid)
X_valid_reduced = selector.transform(X_valid_scaled)

mi_scores = mutual_info_regression(X_train_reduced, y_train, random_state=42)
mi_mask = mi_scores > np.percentile(mi_scores, 20)
X_train_reduced = X_train_reduced[:, mi_mask]
X_valid_reduced = X_valid_reduced[:, mi_mask]

# Optuna tuning for LightGBM
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),  # L2 reg
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'device': 'cpu'
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_reduced, y_train, eval_set=[(X_valid_reduced, y_valid)], callbacks=[lgb.early_stopping(50)])
    preds = model.predict(X_valid_reduced)
    return -pearsonr(y_valid, preds)[0]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("Best Params:", best_params)

# Cross-validation with best params
cv = TimeSeriesSplit(n_splits=5)
cv_scores = []
oof_lgbm = np.zeros(len(y))  # For blending later

for train_idx, valid_idx in cv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_tr_reduced = selector.fit_transform(X_tr_scaled)[:, mi_mask]
    X_val_scaled = scaler.transform(X_val)
    X_val_reduced = selector.transform(X_val_scaled)[:, mi_mask]
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_tr_reduced, y_tr, eval_set=[(X_val_reduced, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    
    preds = model.predict(X_val_reduced)
    corr = pearsonr(y_val, preds)[0]
    cv_scores.append(corr)
    oof_lgbm[valid_idx] = preds

print("CV Pearson Scores:", cv_scores)
print("Mean CV Pearson:", np.mean(cv_scores))

# Retrain on full data
X_full_scaled = scaler.fit_transform(X)
X_full_reduced = selector.transform(X_full_scaled)[:, mi_mask]

model = lgb.LGBMRegressor(**best_params)
model.fit(X_full_reduced, y)

# Predict on test
X_test_feat = test_df.drop(columns=['id', 'label'], errors='ignore')
X_test_scaled = scaler.transform(X_test_feat)
X_test_reduced = selector.transform(X_test_scaled)[:, mi_mask]
test_preds = model.predict(X_test_reduced)
test_preds = np.clip(test_preds, y.quantile(0.01), y.quantile(0.99))

# Submission
ids = test_df['id'] if 'id' in test_df.columns else range(len(test_preds))
submission = pd.DataFrame({"id": ids, "prediction": test_preds})
submission.to_csv("lgbm_submission.csv", index=False)
print("✅ LGBM Submission saved to lgbm_submission.csv")

# For blending with NN (after running NN code)
# nn_df = pd.read_csv('nn_submission.csv')
# blended_preds = 0.8 * submission['prediction'] + 0.2 * nn_df['prediction']
# blended_submission = pd.DataFrame({"id": ids, "prediction": blended_preds})
# blended_submission.to_csv("blended_submission.csv", index=False)