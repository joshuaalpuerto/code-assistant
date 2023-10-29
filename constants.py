import os

# from dotenv import load_dotenv
from chromadb.config import Settings

current_file = __file__ if "__file__" in locals() else os.path.abspath(sys.argv[0])
PACKAGES_DIRECTORY_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(current_file))
)

CODE_ASSISTANT_PATH = os.path.dirname(os.path.realpath(__file__))

PERSIST_DIRECTORY = f"{CODE_ASSISTANT_PATH}/DB"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    # We are using default values here
    chroma_db_impl="duckdb+parquet",
    anonymized_telemetry=False,
    is_persistent=True,
)
