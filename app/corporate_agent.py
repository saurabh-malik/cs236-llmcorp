import torch
from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList
import transformers
from langchain.llms import HuggingFacePipeline

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

model_id = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '#####################'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

#Todo - MOve to utils
# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

#res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse.")
query = "Whos is the author of '2D excitation information by MPS method on infinite helixes' paper?"
llm = HuggingFacePipeline(pipeline=generate_text)
res = llm(prompt=query)
print('----------------------Response from agent powered by Naive LLM-------------------------')
print(res)


######Initiate Vector DB
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

papers_vectorstore = FAISS.load_local("faiss_papers_index", embeddings)

######Initializing Chain
chain = ConversationalRetrievalChain.from_llm(llm, papers_vectorstore.as_retriever(), return_source_documents=True)

chat_history = []

result = chain({"question": query, "chat_history": chat_history})
print('-----------------Response from agent powered by RAG-enhanced Llama-2 model--------------------')
print(result['answer'])