from langchain.chains import ConversationalRetrievalChain


class MyConversationalRetrievalChain(ConversationalRetrievalChain):
    @classmethod
    def from_llm(cls, llm, retriever, return_source_documents=True):
        chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=return_source_documents)
        return chain
