import torch
from model.customchain.chains import MyConversationalRetrievalChain
from model.utils.setup_utils import get_llm, get_vector_db


# Define a service that uses dependencies
def get_llm_answer(question):
    llm = get_llm()  # Obtain the MyHuggingFacePipeline object from the dependency
    vector_db = get_vector_db()  # Obtain the MyFAISS object from the dependency

    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

    chain = MyConversationalRetrievalChain.from_llm(llm.pipeline, vector_db.as_retriever(), return_source_documents=True)
    chat_history = []

    result = chain({"question": question, "chat_history": chat_history})

    return result