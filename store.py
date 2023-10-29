from langchain.vectorstores import Chroma
import utils

from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY


def generate_document_ids(documents):
    """
    current dcouments source is absolute path
    we need only the the relative path from the git repo
    """

    repo_folder_path = utils.get_main_repo_folder() + "/"

    document_ids = [
        doc.metadata["source"].replace(repo_folder_path, "") for doc in documents
    ]

    return document_ids


def create_codebase_embeddings(documents, embedding_function):
    """
    Remember we don't need to provide the collection name as we will use langchain default
    """
    return Chroma.from_documents(
        documents,
        embedding_function,
        ids=generate_document_ids(documents),
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
