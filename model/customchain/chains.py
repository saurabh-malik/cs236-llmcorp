from langchain.chains import ConversationalRetrievalChain

class MyConversationalRetrievalChain:
    @classmethod
    def from_llm(cls, llm, retriever, return_source_documents):
        chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
        return chain