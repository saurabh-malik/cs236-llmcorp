from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader


def get_default_embedding_model():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    return embeddings


def get_document_splits(folder_dir):
    loader = DirectoryLoader(folder_dir, glob="**/*.pdf", show_progress=True, use_multithreading=True)
    documents = loader.load()
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)
    return all_splits, documents


def index_documents(folder_dir: str, index_save_dir: str):
    all_splits, documents = get_document_splits(folder_dir)

    embeddings = get_default_embedding_model()

    # storing embeddings in the vector store
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    vectorstore.save_local(index_save_dir)

    return len(documents)


def add_to_existing_index(current_index_dir: str, new_doc_folder_dir: str, index_save_dir: str = None):
    # Load current index
    embeddings = get_default_embedding_model()
    db = FAISS.load_local(current_index_dir, embeddings)

    # Index new documents
    all_splits, documents = get_document_splits(new_doc_folder_dir)
    new_db = FAISS.from_documents(all_splits, embeddings)

    # Merge dbs
    db.merge_from(new_db)
    db.save_local(index_save_dir if index_save_dir else current_index_dir)

    return len(documents)

def index_document(current_index_dir: str, file: str):
    # Load current index
    embeddings = get_default_embedding_model()
    db = FAISS.load_local(current_index_dir, embeddings)

    # Index new documents
    loader = PyPDFLoader(file)
    texts = loader.load()
    print(len(texts))
    print(texts)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    splits = text_splitter.split_documents(texts)
    print(len(splits))
    print(splits)
    new_db = FAISS.from_documents(splits, embeddings)
    print("New Document Indexed")

    # Merge dbs
    db.merge_from(new_db)
    db.save_local(current_index_dir)
    print("DB Merged and Saved")



def reset_Index(current_index_dir: str, baseline_index_dir: str):
    embeddings = get_default_embedding_model()
    db = FAISS.load_local(baseline_index_dir, embeddings)
    db.save_local(current_index_dir)

def index_web_content(index_dir, documents):
    #Get Default Embeddings
    embeddings = get_default_embedding_model()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(documents)
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(index_dir)
    db.save_local(index_dir+"_baseline")