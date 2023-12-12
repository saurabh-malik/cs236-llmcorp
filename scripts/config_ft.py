class FTConfig:
	# @title Configurations

	################################################################################
	# HuggingFace
	################################################################################

	hf_auth = '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'
	hf_write_token = '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'

	################################################################################
	# QLoRA parameters
	################################################################################

	lora_r = 32 # Defines the size or dimensionality of the vectors used within the attention computation. A higher value can capture more intricate patterns but might be computationally expensive and cause overfitting.
	lora_alpha = 16 # Helps in managing the magnitude of values.
	lora_dropout = 0.2 # 20% of the neurons in LoRA layers would be turned off at each training step.

	################################################################################
	# bitsandbytes parameters
	################################################################################

	use_4bit = True # Reducing to 4-bit precision instead of 32.
	bnb_4bit_compute_dtype = "float16" # Less precise than float32 but allows computations to be faster.
	bnb_4bit_quant_type = "nf4" # A specific 4-bit quantization technique
	use_nested_quant = False # Activate nested quantization for 4-bit base models (double quantization)

	################################################################################
	# TrainingArguments parameters
	################################################################################

	num_train_epochs = 1 # number of full passes through the training data
	# Enable fp16/bf16 training (set bf16 to True with an A100) used to speed up training and reduce memory usage
	fp16 = False
	bf16 = False
	# Batch size per GPU for training and evaluation
	per_device_train_batch_size = 1 #SM - Chagned to 1 from 4 due to limited compute
	per_device_eval_batch_size = 1 #SM - Chagned to 1 from 4 due to limited compute
	# Number of update steps to accumulate the gradients for
	gradient_accumulation_steps = 4 #SM - Chagned to 4 from 4 due to limited compute
	# Enable gradient checkpointing
	gradient_checkpointing = True
	# Maximum gradient normal (gradient clipping)
	max_grad_norm = 0.3
	# Initial learning rate (AdamW optimizer)
	learning_rate = 1e-5
	# Weight decay to apply to all layers except bias/LayerNorm weights
	weight_decay = 0.001
	# Optimizer to use
	optim = "paged_adamw_32bit"
	# Learning rate schedule
	lr_scheduler_type = "cosine"
	# Number of training steps (overrides num_train_epochs)
	max_steps = -1
	# Ratio of steps for a linear warmup (from 0 to learning rate)
	warmup_ratio = 0.03
	# Group sequences into batches with same length
	group_by_length = True
	# Save checkpoint every X updates steps
	save_steps = 0
	# Log every X updates steps
	logging_steps = 25


	################################################################################
	# SFT parameters
	################################################################################

	# Maximum sequence length to use
	max_seq_length = 512 #SM - Changed due to limited Compute
	# Pack multiple short examples in the same input sequence to increase efficiency
	packing = False
	# Load the entire model on the GPU 0
	device_map = {"": 0}

config = FTConfig()