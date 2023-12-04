import os
import json
from urllib.parse import urlparse
from langchain.utilities import ApifyWrapper
from langchain.document_loaders import ApifyDatasetLoader
from langchain.document_loaders.base import Document
from config import config

#Load Apify
os.environ["APIFY_API_TOKEN"] = config.crwaler_key
apify = ApifyWrapper()

#To save compute and time, don't crawl again and again
crawl_history_file = 'crawl_history.json'

def load_crawl_history():
    if os.path.exists(crawl_history_file):
        with open(crawl_history_file, 'r') as file:
            return json.load(file)
    return {}

def save_crawl_history(history):
    with open(crawl_history_file, 'w') as file:
        json.dump(history, file)

def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def filter_urls(dataset_item):
    excluded_urls = ["https://www.globallogic.com/hr/", "https://www.globallogic.com/jp/", "https://www.globallogic.com/de/", "https://www.globallogic.com/in/", "https://www.globallogic.com/il/", "https://www.globallogic.com/latam/", "https://www.globallogic.com/se/", "https://www.globallogic.com/pl/", "https://www.globallogic.com/ro/", "https://www.globallogic.com/sk/","https://www.globallogic.com/ua/","https://www.globallogic.com/uk/",]
    url = dataset_item.get("url", "")
    if not any(excluded_url in url for excluded_url in excluded_urls):
        return Document(
            page_content=dataset_item.get("text", "") or "", 
            metadata={"source": url}
        )
    return None

def crawl_website(url: str):
	history = load_crawl_history()
	domain = get_domain(url)

	if domain in history:
		dataset_id = history[domain]
		loader = ApifyDatasetLoader(
			dataset_id=dataset_id,
			dataset_mapping_function=filter_urls
		)
	else:
		if config.isCrawlingOpen:
			# Call the Actor to obtain text from the crawled webpages.
			#ToDo - SM include excluded_urls
			urls = [{"url": url}]
			loader = apify.call_actor(
				actor_id="apify/website-content-crawler",
				run_input={
			    	"startUrls": urls
				},
				dataset_mapping_function=lambda item: Document(
			    	page_content=item["text"] or "", metadata={"source": item["url"]}
				),
			)
		else:
			raise Exception("Crawling for new websites is closed")
	documents = loader.load()
	documents = [document for document in documents if document is not None]
	return documents