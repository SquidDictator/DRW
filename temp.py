# Load data
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import root_mean_squared_error
from scipy.stats import rankdata, pearsonr

train_df = pd.read_parquet("DRW/train.parquet")
test_df = pd.read_parquet("DRW/test.parquet")

# Select features
feature_cols = [col for col in train_df.columns if col.startswith("X")]
X = train_df[feature_cols].copy()
y = train_df['label'].shift(-1).iloc[:-1].copy()
X = X.iloc[:-1, :].copy()

# === Feature Engineering ===
epsilon = 1e-5
X['log_abs_752'] = np.log(np.abs(X['X752']) + epsilon)
X['log_abs_331'] = np.log(np.abs(X['X331']) + epsilon)
X['ratio_752_331'] = X['X752'] / (X['X331'] + epsilon)
X['ratio_331_752'] = X['X331'] / (X['X752'] + epsilon)
X['squared_752'] = X['X752'] ** 2
X['squared_331'] = X['X331'] ** 2
X['sin_752'] = np.sin(X['X752'])
X['cos_331'] = np.cos(X['X331'])

# Test set: same features
for df in [test_df]:
    df['log_abs_752'] = np.log(np.abs(df['X752']) + epsilon)
    df['log_abs_331'] = np.log(np.abs(df['X331']) + epsilon)
    df['ratio_752_331'] = df['X752'] / (df['X331'] + epsilon)
    df['ratio_331_752'] = df['X331'] / (df['X752'] + epsilon)
    df['squared_752'] = df['X752'] ** 2
    df['squared_331'] = df['X331'] ** 2
    df['sin_752'] = np.sin(df['X752'])
    df['cos_331'] = np.cos(df['X331'])

# Clustering
top_feats = ['X752', 'X331']
X_cluster = X[top_feats]
X_scaled = StandardScaler().fit_transform(X_cluster)
kmeans = KMeans(n_clusters=5, random_state=42).fit(X_scaled)
X['cluster'] = kmeans.labels_

# Cluster analysis
cluster_means = pd.DataFrame({'label': y, 'cluster': X['cluster']}).groupby('cluster')['label'].mean()
print(cluster_means)

# Train/val split
split_idx = int(0.8 * len(X))
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

# Train model
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[early_stopping(50), log_evaluation(100)]
)
print("✅ Best iteration:", model.best_iteration_)

# SHAP analysis
explainer = shap.Explainer(model)
shap_values = explainer(X_val)
shap.summary_plot(shap_values, X_val, max_display=20)

# === Cluster-based validation predictions
X_val['cluster'] = kmeans.predict(StandardScaler().fit_transform(X_val[top_feats]))
X_val['pred'] = X_val['cluster'].map(cluster_means)
pcc = pearsonr(y_val, X_val['pred'])[0]
rmse = root_mean_squared_error(y_val, X_val['pred'])
print("Holdout PCC:", pcc)
print("Holdout RMSE:", rmse)

# === Submission
X_test_scaled = StandardScaler().fit_transform(test_df[top_feats])
test_df['cluster'] = kmeans.predict(X_test_scaled)
test_df['pred'] = test_df['cluster'].map(cluster_means)
test_df['target'] = rankdata(test_df['pred'])

submission = pd.DataFrame({
    "ID": test_df.index,
    "prediction": test_df['target']
})
submission.to_csv("s.csv", index=False)
print("✅ Submission saved as s.csv")
