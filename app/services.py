import os
import torch
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage

from model.customchain.chains import MyConversationalRetrievalChain
from model.utils.setup_utils import get_llm, resetVectorIndex, get_vector_db
from model.utils import index_utils
from config import config

#ToDo SM (1) - Replace the custom chains with Runnables
#ToDo SM (2) - Maintain chat history


#Prompt for QA Agent
qa_system_prompt = """<s>[INST] <<SYS>>
You are Corpy, an AI based agent of Dumdum Dummy Inc organization, trained on  private data to ans the questions. Use the following pieces of context to answer the question at the end. 
If you don't know the answer or if there is no context, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
<</SYS>>
Question: {question} [/INST]
Helpful Answer:"""

rag_prompt_custom = PromptTemplate.from_template(qa_system_prompt)

#Prompt for Standalone Question Generation :: Keeping it simple for now.
saq_base_template = (
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)
saq_base_prompt = PromptTemplate.from_template(saq_base_template)


# Define a service that uses dependencies
def get_llm_answer(question):
    llm = get_llm()  # Obtain the MyHuggingFacePipeline object from the dependency
    vector_db = get_vector_db()  # Obtain the MyFAISS object from the dependency

    # Keeping default search behavior for retriver
    # ToDo SM (3) - Handle Lost in the middle during retrieval

    #retriever = vector_db.as_retriever()
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

   
    chain = MyConversationalRetrievalChain.from_llm(llm.pipeline, retriever, saq_base_prompt, rag_prompt_custom, return_source_documents=True)
    chat_history = []

    result = chain({"question": question, "chat_history": chat_history})
    print(result)

    return result

# Index new file
def process_and_index_file(fileName):
    #Index the file
    index_utils.index_document(config.kb_index,fileName)

    #Reset vector Index
    resetVectorIndex()
    print("DB Reloaded")

    #File Deleted
    delete_file(fileName)

def delete_file(file_path: str):
    """ Delete a file from the filesystem. """
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    else:
        print(f"The file {file_path} does not exist.")
        return False

