import torch
import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria
)
from langchain.llms import HuggingFacePipeline
from config import config

class MyHuggingFacePipeline:
    def __init__(self, model_id, hf_auth, stopping_criteria):
        self.model = self.load_model(model_id, hf_auth)
        self.pipeline = self.create_pipeline(model_id, hf_auth, stopping_criteria)

    def load_model(self, model_id, hf_auth):
        
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

        with torch.no_grad():
            # set quantization configuration to load a large model with less GPU memory
            # this requires the `bitsandbytes` library
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model_config = AutoConfig.from_pretrained(
                model_id,
                use_auth_token=hf_auth
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                config=model_config,
                quantization_config=bnb_config,
                device_map='auto',
                use_auth_token=hf_auth
            )

            # enable evaluation mode to allow model inference
            model.eval()

            return model

    def create_pipeline(self, model_id, hf_auth, stopping_criteria):
        # Implement any additional pipeline configuration logic if needed
        #SM - ToDo: This needs further refactoring.
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=stopping_criteria,
            temperature=0.1,
            max_new_tokens=512,
            repetition_penalty=1.1
        )
        return HuggingFacePipeline(pipeline=pipeline)

    def prompt(self, text):
        # SM ToDO: Need to implement later for now just simple prompt, without chat history
        prompt = {
            "question": text,
            "chat_history": []
        }
        # Pass the prompt dictionary to the LLM model
        response = self.pipeline(prompt=prompt)

        #SM ToDO - Also need to do post-processing later to solve long context.
        # response = post_processing(response)
        return response
