import torch
from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList
import transformers
from langchain.llms import HuggingFacePipeline
import argparse

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--kb', type=str, default='faiss_200_papers_index', help="Vector DB Index for your knowledge base")

args = parser.parse_args()


model_id = 'meta-llama/Llama-2-13b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

with torch.no_grad():
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, you need an access token
    hf_auth = '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'
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

    llm = HuggingFacePipeline(pipeline=generate_text)

    print("###############################################################")

    query_1 = "Whos is the author of 'A Near-Quadratic Sample Complexity Reduction for Agnostic Learning via Quantum Algorithms' paper?"
    print("Query-1: {}".format(query_1))
    print('----------------------Query-1 ans from agent powered by Naive Llama-2 model-------------------------')
    res = llm(prompt=query_1)
    print(res)


    ######Initiate Vector DB
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    papers_vectorstore = FAISS.load_local(args.kb, embeddings)

    ######Initializing Chain
    chain = ConversationalRetrievalChain.from_llm(llm, papers_vectorstore.as_retriever(), return_source_documents=True)

    chat_history = []

    print('-----------------Query-1 ans from agent powered by RAG-enhanced Llama-2 model--------------------')
    result = chain({"question": query_1, "chat_history": chat_history})
    print(result['answer'])

    print("###############################################################")
    query_2 = "Where are the source of valley-polarized electron as per the paper: 'A ballistic electron source with magnetically-controlled valley polarization in bilayer graphene' paper."
    print("Query-2: {}".format(query_2))
    print('----------------------Query-2 ans from agent powered by Naive LLM-------------------------')
    res = llm(prompt=query_2)
    print(res)

    print('-----------------Query-2 ans from agent powered by RAG-enhanced Llama-2 model--------------------')
    result = chain({"question": query_2, "chat_history": chat_history})
    print(result['answer'])