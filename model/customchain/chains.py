from langchain.chains import ConversationalRetrievalChain

class MyConversationalRetrievalChain(ConversationalRetrievalChain):
    @classmethod
    def from_llm(cls, llm, retriever, condense_q_prompt, prompt, return_source_documents=True):
        #chain = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt = condense_q_prompt, return_source_documents=return_source_documents, combine_docs_chain_kwargs={'prompt': prompt})
        chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=return_source_documents, combine_docs_chain_kwargs={'prompt': prompt})
        return chain
