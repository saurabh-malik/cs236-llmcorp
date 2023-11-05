from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import glob


PAPER_DIR = "20-papers"


##Load the pdf documents from data
loader = DirectoryLoader(f'./data/papers/{PAPER_DIR}/', glob="**/*.pdf", show_progress=True, use_multithreading=True)
documents = loader.load()
print("Total Files loaded: {}".format(len(documents)))


# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

vectorstore.save_local(f"faiss_{PAPER_DIR.replace('-', '_')}_index")