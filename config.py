class AppConfig:
    kb_index = 'faiss_web_index'
    kb_index_baseline = 'faiss_web_index_baseline'
    model_id = 'meta-llama/Llama-2-13b-chat-hf'
    #model_id = 'meta-llama/Llama-2-7b-chat-hf'
    embedding_model = 'sentence-transformers/all-mpnet-base-v2'
    hf_auth = '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'
    crwaler_key = 'apify_api_MgtGtJKQA1NLYDV5rmnivmYx6kphkp1p57LV'
    isCrawlingOpen = False
    #Add other configuration settings here

config = AppConfig()
