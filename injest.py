import logging
import argparse

from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import store

from constants import (
    PACKAGES_DIRECTORY_PATH,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)


def load_codebase(directory):
    loader = GenericLoader.from_filesystem(
        directory,
        # Includes both root and subdirectories
        glob="**/*",
        exclude=[
            f"{directory}/node_modules/*",
            f"{directory}/node_modules/**/*",
            f"{directory}stories/*",
            f"{directory}/stories/**/*",
            f"{directory}.next/*",
            f"{directory}/.next/**/*",
            f"{directory}.storybook/*",
            f"{directory}/.storybook/**/*",
            f"{directory}k8s/*",
            f"{directory}/k8s/**/*",
        ],
        suffixes=[".js"],
        show_progress=True,
        parser=LanguageParser(language=Language.JS, parser_threshold=500),
    )
    documents = loader.load()
    return documents


def split_documents(documents):
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=2000, chunk_overlap=200
    )
    return js_splitter.split_documents(documents)


def main():
    """
    Implements loading codebase directory and creating AST using LanguageParser.
    Then it will split the file to reasonable chunk for for embeddings.

    NOTE: This might take time for large codebase
    """

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="A simple Python script will injest codebase."
    )

    # Add arguments
    parser.add_argument(
        "--package",
        help="Path to the package folder, remember when searching you need to pass this also.",
        required=True,
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    package = f"/{args.package}"
    CODE_BASE_PATH = PACKAGES_DIRECTORY_PATH + package
    DB_FULL_DIRECTORY = PERSIST_DIRECTORY + package

    documents = load_codebase(CODE_BASE_PATH)

    texts = split_documents(documents)

    logging.info(f"Loaded {len(documents)} documents from {CODE_BASE_PATH}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store.create_codebase_embeddings(documents, embedding_function)

    # db = FAISS.from_documents(texts, embeddings)
    # db.save_local(DB_FULL_DIRECTORY)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
