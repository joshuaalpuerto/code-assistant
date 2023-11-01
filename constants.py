import os
from dotenv import load_dotenv

load_dotenv()

CODE_ASSISTANT_PATH = os.path.dirname(os.path.realpath(__file__))

PERSIST_DIRECTORY = f"{CODE_ASSISTANT_PATH}/db"
REPO_PATH = os.getenv("REPO_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME") or "BAAI/bge-large-en-v1.5"
MODEL_ID = os.getenv("MODEL_ID") or "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_FILE = (
    os.getenv("MODEL_ID") or "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)  # medium, balanced quality - recommended
MODEL_TYPE = os.getenv("MODEL_TYPE") or "mistral"

# users/.models
MODELS_PATH = "./models"
