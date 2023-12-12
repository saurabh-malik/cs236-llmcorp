from langchain.vectorstores.faiss import FAISS
from langchain.utilities import ApifyWrapper
from langchain.document_loaders import ApifyDatasetLoader
from langchain.document_loaders.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

apify = ApifyWrapper()

urls = [{"url": "https://www.globallogic.com/"}]
# Call the Actor to obtain text from the crawled webpages
'''loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": urls
    },
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

documents = loader.load()
'''

def filter_urls(dataset_item):
    excluded_urls = ["https://www.globallogic.com/hr/", "https://www.globallogic.com/jp/", "https://www.globallogic.com/de/", "https://www.globallogic.com/in/", "https://www.globallogic.com/il/", "https://www.globallogic.com/latam/", "https://www.globallogic.com/se/", "https://www.globallogic.com/pl/", "https://www.globallogic.com/ro/", "https://www.globallogic.com/sk/","https://www.globallogic.com/ua/","https://www.globallogic.com/uk/",]
    url = dataset_item.get("url", "")
    if not any(excluded_url in url for excluded_url in excluded_urls):
        return Document(
            page_content=dataset_item.get("text", "") or "", 
            metadata={"source": url}
        )
    return None

loader = ApifyDatasetLoader(
    dataset_id="LwetriLhnHFVCtBdR",
    dataset_mapping_function=filter_urls
)
texts = loader.load()

texts = [text for text in texts if text is not None]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
splits = text_splitter.split_documents(texts)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss_web_index")