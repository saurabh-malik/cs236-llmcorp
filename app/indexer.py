from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


##Load the pdf documents
#for _ in files:
loader = PyPDFLoader("./data/papers/papers/2D excitation information by MPS method on infinite helixes.pdf")
documents = loader.load()


# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

vectorstore.save_local("faiss_papers_index")