from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever
from langchain.schema.language_model import BaseLanguageModel


class MyConversationalRetrievalChain(ConversationalRetrievalChain):
    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, retriever: BaseRetriever, condense_q_prompt=None, prompt=None,
                 return_source_documents=True):
        if condense_q_prompt:
            chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever,
                condense_question_prompt=condense_q_prompt,
                combine_docs_chain_kwargs={'prompt': prompt},
                return_source_documents=return_source_documents,
            )
        else:
            chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever,
                return_source_documents=return_source_documents,
            )
        #chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=return_source_documents, combine_docs_chain_kwargs={'prompt': prompt})
        return chain
