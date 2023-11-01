import logging

import langchain
from langchain.llms import CTransformers
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain import PromptTemplate

from constants import MODEL_ID, MODEL_FILE, MODEL_TYPE
from prompts import PROMPT_EXTRACTOR


# Enable debug
langchain.debug = True


def load():
    """
    follow https://python.langchain.com/docs/integrations/providers/ctransformers
    """

    logging.info(f"Loading Model: {MODEL_ID}, on: cpu")
    logging.info("This action can take a few minutes!")
    config = {
        "max_new_tokens": 2048,
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
    }

    llm = CTransformers(
        model=MODEL_ID,
        model_file=MODEL_FILE,
        model_type=MODEL_TYPE,
        config=config,
    )

    return llm


def call(llm, query, context, prompt):
    # Log genereated info
    handler = StdOutCallbackHandler()
    # We use LLMChain + manual retrieval because the answers for our question is inside metadata.
    # There no easyway to do it in Langchain.
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True, callbacks=[handler])

    return llm_chain.predict(question=query, context=context)


def llm_chain_extractor(llm, query, context):
    compressor = LLMChainExtractor.from_llm(
        llm,
        prompt=PromptTemplate(
            template=PROMPT_EXTRACTOR, input_variables=["question", "context"]
        ),
    )
    return compressor.llm_chain.predict_and_parse(
        question=query,
        context=context,
    )
