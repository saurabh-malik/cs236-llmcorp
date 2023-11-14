from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import glob

from model.utils.index_utils import index_documents

PAPER_DIR = "20-papers"


##Load the pdf documents from data
doc_count = index_documents(
    f'./data/papers/{PAPER_DIR}/',
    f"faiss_{PAPER_DIR.replace('-', '_')}_index"
)
print(f"Total Files loaded: {doc_count}")