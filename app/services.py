import torch
from fastapi import Depends
from customchain.chains import MyConversationalRetrievalChain
from config import config
from app.dependencies import get_llm, get_vector_db
from customchain.llms import MyHuggingFacePipeline
from customchain.vectorstores import MyFAISS

# Define a service that uses dependencies
def get_llm_answer(question):
    llm = get_llm()  # Obtain the MyHuggingFacePipeline object from the dependency
    vector_db = get_vector_db()  # Obtain the MyFAISS object from the dependency

    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

    chain = MyConversationalRetrievalChain.from_llm(llm.pipeline, vector_db.as_retriever(), return_source_documents=True)
    chat_history = []

    result = chain({"question": question, "chat_history": chat_history})

    return result