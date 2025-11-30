from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
import pandas as pd
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
# HuggingFace dataset path
DATASET_PATH = "hf://datasets/Parthipan00410/Bank-Customer-Churn-Data/bank_customer_churn.csv"
data=pd.read_csv(DATASET_PATH)
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

# Categorical features
categorical_features = ["Geography"]

X=data[numeric_features+categorical_features]
Y=data[target]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
X_train.to_csv('X_train.csv',index=False)
X_test.to_csv('X_test.csv',index=False)
Y_train.to_csv('Y_train.csv',index=False)
Y_test.to_csv('Y_test.csv',index=False)

# Upload each file to HuggingFace dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for i in files:
  api.upload_file(
      path_or_fileobj=i,
      path_in_repo=os.path.basename(i),
      repo_id = "Parthipan00410/Bank-Customer-Churn-Data",
repo_type = "dataset"

  )
