import os
from dotenv import load_dotenv

load_dotenv()

CODE_ASSISTANT_PATH = os.path.dirname(os.path.realpath(__file__))

PERSIST_DIRECTORY = f"{CODE_ASSISTANT_PATH}/db"
REPO_PATH = os.getenv("REPO_PATH")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME") or "BAAI/bge-large-en-v1.5"
