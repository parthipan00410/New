import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# HF API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset paths
DATA_REPO = "Parthipan00410/Bank-Customer-Churn-Data"

Xtrain = pd.read_csv(f"hf://datasets/{DATA_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATA_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATA_REPO}/ytrain.csv")
ytest = pd.read_csv(f"hf://datasets/{DATA_REPO}/ytest.csv")

# Class weight for imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
target='Exited'
# Numerical features
numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)
# XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1,0.11],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}
model=make_pipeline(preprocessor,xgb_model)
model_gv=GridSearchCV(model,param_grid,cv=5,n_jobs=-1)
model_gv.fit(X_train,Y_train)
best_model = model_gv.best_estimator_
# Predictions
threshold = 0.45
train_proba = best_model.predict_proba(Xtrain)[:, 1]
test_proba = best_model.predict_proba(Xtest)[:, 1]

train_pred = (train_proba >= threshold).astype(int)
test_pred = (test_proba >= threshold).astype(int)

print("Train classification report:")
print(classification_report(ytrain, train_pred))

print("Test classification report:")
print(classification_report(ytest, test_pred))

# Save model
joblib.dump(best_model, "best_churn_model.joblib")
print("Model saved.")

# HF Model Repo
repo_id = "Parthipan00410/Bank-Customer-model"
repo_type = "model"

# Create model repo if not exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("Model repo exists.")
except RepositoryNotFoundError:
    print("Model repo not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# Upload model
api.upload_file(
    path_or_fileobj="best_churn_model.joblib",
    path_in_repo="best_churn_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Model uploaded successfully.")

