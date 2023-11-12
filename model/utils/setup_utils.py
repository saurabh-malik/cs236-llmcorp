import torch
from model.customchain.llms import MyHuggingFacePipeline
from model.customchain.vectorstores import MyFAISS
from config import config
from langchain.embeddings import HuggingFaceEmbeddings
from model.utils.stopping_criteria import StopOnTokens
from transformers import StoppingCriteriaList
import transformers

llm = None
vector_db = None

# Dependency to initialize LLM model
def get_llm():
    global llm
    if llm is None:
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_id,
            use_auth_token=config.hf_auth
        )
        stop_list = ['\nHuman:', '\n```\n']
        stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        llm = MyHuggingFacePipeline(
            model_id=config.model_id,
            hf_auth=config.hf_auth,
            stopping_criteria=stopping_criteria
        )
    return llm

# Dependency to initialize vector database
def get_vector_db():
    global vector_db
    if vector_db is None:
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": device}
        )

        vector_db = MyFAISS.load_local(config.kb_index, embeddings)
    return vector_db