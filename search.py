import logging
import argparse
import store
import model

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
)


# Naive top_k retriever (there a chance it gets incorrect context)
def get_documents_as_context(documnets):
    inputs = [f"- {doc.page_content}" for doc in documnets]
    context = "\n".join(inputs)
    return context


def get_relevant_documents(query, args):
    target_folder = f"/{args.target_folder}"

    DB_FULL_DIRECTORY = PERSIST_DIRECTORY + target_folder

    embedding = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # So we can use cos_sim
    )

    db = store.get_db(persist_directory=DB_FULL_DIRECTORY, embedding=embedding)
    retriever = db.as_retriever(
        # It will find similar documents, then iteratively remove almost duplicate results.
        # This return diverse result
        search_type="mmr",
        search_kwargs={"k": 10},
    )

    # We shouldn't rely with score because if the users query is short it will have still high relevance
    return retriever.get_relevant_documents(query)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="A simple Python script that will search codebase base on your query."
    )

    # Add arguments
    parser.add_argument(
        "--target_folder", help="Package you apply when ingesting.", required=True
    )
    parser.add_argument("--query", help="Question.", required=True)

    # Parse the command-line arguments
    args = parser.parse_args()
    query = args.query

    llm = model.load()

    context = get_relevant_documents(query, args)
    # format the string
    context = get_documents_as_context(context)
    result = model.llm_chain_extractor(llm, query, context)

    print("\n\n> Question:")
    print(query)
    print("-" * 20)
    print(result)

    # # Print the relevant sources used for the answer for FAISS
    # print(
    #     "----------------------------------SOURCE DOCUMENTS---------------------------"
    # )
    # for document in documents:
    #     print("\n> " + "{doc_source}".format(doc_source=document.metadata["source"]))
    #     print(document.metadata)
    #     print("-" * 20)
    # print(
    #     "----------------------------------SOURCE DOCUMENTS---------------------------"
    # )
