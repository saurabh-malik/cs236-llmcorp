import os
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset
import torch
from config_ft import config

model_name = "meta-llama/Llama-2-13b-chat-hf"
new_model = "model/fine-tuned/llama-2-13b-CorpGL"
output_dir = "./results"
hf_token = "your_huggingface_token"

#def apply_lora_to_model(model):
    # Custom function to apply LoRA modifications
    # This will involve manually adjusting the model's layers and parameters
    # The implementation details will depend on the model's architecture
    # ...

def main():

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply LoRA modifications to the model
    #apply_lora_to_model(model)

    # Load and preprocess your dataset
    # ...
    dataset = load_dataset('csv', data_files='./data/corporate/finetune_data.csv', split='train')

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and config.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=config.device_map,
        use_auth_token=config.hf_auth
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=config.hf_auth, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field='formatted_text',
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config.packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)

    ## Ignore warnings
    logging.set_verbosity(logging.CRITICAL)
    
    
    # Run text generation pipeline with our next model
    question = "What capabilities does GlobalLogic offer to help businesses turn Big Data into usable knowledge?"
    prompt = """
    <s>[INST] <<SYS>>
    Your name is Corpy, an AI-based agent from GlobalLogic Inc. Your role is to answer the questions specifically related to GlobalLogic.
    <</SYS>>
    Question: What capabilities does GlobalLogic offer to help businesses turn Big Data into usable knowledge? [/INST]
    Helpful Answer:
    """
    
    pipe = pipeline(task="text-generation", model=trainer.model, tokenizer=tokenizer, max_length=512)
    #result = pipe(f"<s>[INST]<<SYS>> {context} <</SYS>> {prompt} [/INST]")
    result = pipe(f"<s>[INST]{prompt} [/INST]")
    print(result[0]['generated_text'])

    #Push To HuggingFace
    #model.push_to_hub(new_model, use_temp_dir=False, use_auth_token=config.hf_write_token)
    #tokenizer.push_to_hub(new_model, use_temp_dir=False, use_auth_token=config.hf_write_token)

if __name__ == "__main__":
    main()
