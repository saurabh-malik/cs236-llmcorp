from langchain.vectorstores import FAISS

class MyFAISS:
    @staticmethod
    def load_local(kb_index, embeddings):
        return FAISS.load_local(kb_index, embeddings)
