from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload your Streamlit app folder to your HuggingFace Space
api.upload_folder(
    folder_path="new/deployment",
    repo_id="Parthipan00410/Bank-Customer-k",  # <-- IMPORTANT
    repo_type="space",
    path_in_repo=""
)
