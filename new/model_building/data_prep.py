from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi
import os
repo_id = "Parthipan00410/Bank-Customer-Churn-Data"
repo_type = "dataset"
# Initialize API client with token
api = HfApi(token=os.getenv("HF_TOKEN"))
try:
  api.repo_info(repo_id=repo_id,repo_type=repo_type)
except RepositoryNotFoundError:
  api.create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
api.upload_folder(folder_path='new/data',repo_id=repo_id,repo_type=repo_type)
