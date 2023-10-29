import logging
import argparse

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)


def get_relevant_documents(query, args):
    package = f"/{args.package}"

    DB_FULL_DIRECTORY = PERSIST_DIRECTORY + package

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # So we can use cos_sim
    )

    db = FAISS.load_local(DB_FULL_DIRECTORY, embeddings)
    retriever = db.as_retriever(
        # It will find similar documents, then iteratively remove almost duplicate results.
        # This return diverse result
        search_type="mmr",
        search_kwargs={"k": 10},
    )

    return retriever.get_relevant_documents(query)


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="A simple Python script that will search codebase base on your query."
    )

    # Add arguments
    parser.add_argument(
        "--package", help="Package you apply when ingesting.", required=True
    )
    parser.add_argument("--query", help="Question.", required=True)

    # Parse the command-line arguments
    args = parser.parse_args()
    query = args.query

    documents = get_relevant_documents(query, args)

    print("\n\n> Question:")
    print(query)

    # Print the relevant sources used for the answer for FAISS
    print(
        "----------------------------------SOURCE DOCUMENTS---------------------------"
    )
    for document in documents:
        print("\n> " + "{doc_source}".format(doc_source=document.metadata["source"]))
        print(document.page_content[:50])
        print("-" * 20)
    print(
        "----------------------------------SOURCE DOCUMENTS---------------------------"
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
