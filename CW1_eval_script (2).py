import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

# Set seed
np.random.seed(123)

# Import training data
train_path = "CW1_train.csv"
test_path = "CW1_test.csv"
if not os.path.exists(train_path):
    # fall back to common downloaded names
    if os.path.exists("CW1_train (1).csv"):
        train_path = "CW1_train (1).csv"
if not os.path.exists(test_path):
    if os.path.exists("CW1_test (2).csv"):
        test_path = "CW1_test (2).csv"

trn = pd.read_csv(train_path)
X_tst = pd.read_csv(test_path)  # This does not include true outcomes (obviously)

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# Identify numeric columns (everything except outcome and categoricals)
outcome_col = "outcome" if "outcome" in trn.columns else trn.columns[0]
numeric_cols = [c for c in trn.columns if c not in categorical_cols + [outcome_col]]

################################################################################
# EDA (quick, focused on outcome vs features)

print("=== EDA SUMMARY ===")
print(f"Train file: {train_path}")
print(f"Rows: {len(trn):,}, Columns: {len(trn.columns)}")
print("Missing values (top 10):")
print(trn.isna().sum().sort_values(ascending=False).head(10))
print("\nOutcome summary:")
print(trn[outcome_col].describe())

# Outcome distribution
plt.figure()
trn[outcome_col].hist(bins=40)
plt.title(f"Outcome distribution: {outcome_col}")
plt.xlabel(outcome_col)
plt.ylabel("Count")

# Correlations with outcome (numeric only)
if numeric_cols:
    corrs = trn[numeric_cols + [outcome_col]].corr(numeric_only=True)[outcome_col]
    corrs = corrs.drop(outcome_col).sort_values(key=lambda s: s.abs(), ascending=False)
    print("\nTop numeric correlations with outcome:")
    print(corrs.head(10))
    plt.figure(figsize=(8, 5))
    corrs.head(15).sort_values().plot(kind="barh")
    plt.title("Top numeric correlations with outcome")
    plt.xlabel("Pearson correlation")

# Outcome by category (mean)
for col in categorical_cols:
    if col in trn.columns:
        grp = trn.groupby(col)[outcome_col].mean().sort_values()
        plt.figure(figsize=(8, 5))
        grp.plot(kind="barh")
        plt.title(f"Mean outcome by {col}")
        plt.xlabel("Mean outcome")

################################################################################
# Model: a stronger regressor with proper preprocessing

X = trn.drop(columns=[outcome_col])
y = trn[outcome_col]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
)

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=400,
    random_state=123,
)

pipe = Pipeline(
    steps=[
        ("prep", preprocess),
        ("model", model),
    ]
)

# Train/validation split for quick performance check
X_trn, X_val, y_trn, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123
)
pipe.fit(X_trn, y_trn)
val_pred = pipe.predict(X_val)
val_r2 = r2_score(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print("\n=== MODEL VALIDATION ===")
print(f"R^2: {val_r2:.4f}")
print(f"RMSE: {val_rmse:.4f}")

# Diagnostic plots
plt.figure()
plt.scatter(y_val, val_pred, s=10, alpha=0.4)
plt.title("Validation: Actual vs Predicted")
plt.xlabel("Actual outcome")
plt.ylabel("Predicted outcome")

plt.figure()
residuals = y_val - val_pred
plt.hist(residuals, bins=40)
plt.title("Validation residuals")
plt.xlabel("Residual")
plt.ylabel("Count")

# Fit on full training data and predict test
pipe.fit(X, y)
yhat = pipe.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat})
out.to_csv('CW1_submission_KNUMBER.csv', index=False) # Please use your k-number here

################################################################################

# At test time, we will use the true outcomes (if available)
true_path = "CW1_test_with_true_outcome.csv"
if os.path.exists(true_path):
    tst = pd.read_csv(true_path)  # You do not have access to this in the real test
    y_tst = tst[outcome_col]
    yhat_tst = pipe.predict(tst.drop(columns=[outcome_col]))
    print("\n=== HELD-OUT TEST (if available) ===")
    print(f"R^2: {r2_score(y_tst, yhat_tst):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_tst, yhat_tst)):.4f}")

# Keep all plots open until you close them
plt.show(block=True)


