import csv
import os.path
import shutil
from typing import Type

import torch
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.faiss import FAISS

from model.customchain.llms import MyHuggingFacePipeline
from model.customchain.vectorstores import MyFAISS
from config import config
from langchain.embeddings import HuggingFaceEmbeddings

from model.utils import index_utils
from model.utils.data_utils import DataSplitGroup, get_paper_names_by_group_idx, get_paper_names_by_split
from model.utils.logging_utils import activate_logger
from model.utils.stopping_criteria import StopOnTokens
from transformers import StoppingCriteriaList
import transformers

llm = None
vector_db = None

logger = activate_logger('setup_utils')


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
        print("Vector DB is NUll")
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": device}
        )

        vector_db = MyFAISS.load_local(config.kb_index, embeddings)
    print("Vector DB is Loaded")
    return vector_db


def get_vector_db_on_split(
        split: DataSplitGroup,
        *,
        vector_store_cls: Type[VectorStore] = FAISS

):
    selected_groups_concat = "-".join([str(group_idx) for group_idx in split.value])
    index_name = f"{vector_store_cls.__name__}_{selected_groups_concat}_papers_index"

    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": device}
    )

    if os.path.isdir(index_name):
        logger.info(f"Reusing pre-generated index {index_name}")
        loaded_vector_db = vector_store_cls.load_local(index_name, embeddings)
    else:
        logger.info(f"Generating new index at {index_name}")

        # Extract papers with specified split to a temp folder
        grouped_paper_folder = os.path.join(os.path.dirname(__file__), f"../../data/papers/{index_name}")
        if os.path.isdir(grouped_paper_folder):
            os.removedirs(grouped_paper_folder)
        os.makedirs(grouped_paper_folder)
        # TODO generate index for selected papers
        for paper_name in get_paper_names_by_split(split):
            shutil.copy(
                os.path.join(os.path.dirname(__file__), f"../../data/papers/1000-papers/{paper_name}.pdf"),
                grouped_paper_folder
            )

        index_utils.index_documents(grouped_paper_folder, index_name)
        loaded_vector_db = vector_store_cls.load_local(index_name, embeddings)

    return loaded_vector_db


def reload_VectorIndex():
    global vector_db
    vector_db = None
    get_vector_db()


CONTEXT = \
f"""
You have no prior knowledge, and need to use utility functions below to retrieve information:

* get_paper_authors(paper_name: str) -> List[str]. 
  The function takes `paper_name` as input parameter and gives a list of strings that are the names of paper authors. If the paper cannot be found, an empty list will be returned.
* get_paper_abstract(paper_name: str) -> List[str].
  The function takes `paper_name` as input parameter and gives the paper abstract as string. If the paper cannot be found, an empty list will be returned.
* You can also use all other python 3 native functions

If you'd like to use them, please embed the function call in your answer like `<begin_call>python_func_call()<end_call>`, and I will evaluate python codes between `<begin_call>` and `<end_call>`. The output of python codes embedded should be a single string.
For example:
Question: Who are the authors of paper "Attention is All You Need"?
Answer: The authors are <begin_call>str(get_paper_authors("Attention is All You Need"))<end_call>.
Question: How many authors are there in paper "Can acoustic early dark energy still resolve the Hubble tension"
Answer: There are <begin_call>str(len(get_paper_authors("Can acoustic early dark energy still resolve the Hubble tension")))<end_call> authors.
Question: Are there even number or odd number of authors in paper "Can acoustic early dark energy still resolve the Hubble tension"
Answer: There are <begin_call>"even" if len(get_paper_authors("Can acoustic early dark energy still resolve the Hubble tension")) % 2 == 0 else "odd"<end_call> authors.
Question: Please summarize paper "Can acoustic early dark energy still resolve the Hubble tension"
Answer: <begin_call>get_paper_abstract("Can acoustic early dark energy still resolve the Hubble tension")<end_call>
"""
