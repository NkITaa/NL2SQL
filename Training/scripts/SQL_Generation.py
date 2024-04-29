import os
import torch
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, logging, Trainer, DataCollatorForSeq2Seq, pipeline
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator, notebook_launcher
from pathlib import Path

################################################################################
# Model label
################################################################################

# Defining the pre-trained model to be used
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Fine-tuned model name
new_model = "SQL-Generation-mistral-7B-v0.1"

# Access token
token = "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"

# Load the entire model on the GPU
device_map = {"" : PartialState().process_index}

# Trust remote code for loading model
trust_remote_code = True

# specifiying whether Cache should be used
use_cache = False

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
load_in_4bit = True

# Activate nested quantization for 4-bit base models (double quantization)
bnb_4bit_use_double_quant = False

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Data type for computation
bnb_4bit_compute_dtype = torch.bfloat16

# Configuring the BitsAndBytes quantization for the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
)

# Initialize tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name, token = token)
# Set pad token to end-of-sequence token
tokenizer.pad_token = "<PAD>"
# Fix weird overflow issue with fp16 training
tokenizer.padding_side = "right"

# Load the pre-trained model for causal language modeling
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token = token,
    device_map = device_map,
    trust_remote_code = trust_remote_code,
    quantization_config = bnb_config,
    use_cache = use_cache,
)

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 32 # I had it on 8 before

# Alpha parameter for LoRA scaling
lora_alpha = 64

# Dropout probability for LoRA layers
lora_dropout = 0.15

# Bias
bias = "none"

# Task type (Causal Language Modeling)
task_type = "CAUSAL_LM"

# Target modules for Lora
target_modules = "all-linear"

# uses Rank-Stabilized LoRA which sets the adapter scaling factor to lora_alpha/math.sqrt(r)
use_rslora = True

# Configuring the LoraConfig for the model
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=bias,
    task_type=task_type,
    target_modules=target_modules,
    use_rslora=use_rslora
)

# prepares model for training (don't know what is done exactly, but it's in QLORA doku)
model = prepare_model_for_kbit_training(model)

# creates peft model
model = get_peft_model(model, peft_config)

# params printing (function only works on peft model)
model.print_trainable_parameters()

def format_instruction_SQL_Generation(sample):
  result = f"""<s>[INST]
    You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
    You must output the SQL query that answers the question. Only Answer with the SQL Query, You are also provided with some suggestions on the columns to use in the Schema Link Section
    ### Question:
    {sample['question']}
    ### Schema:{sample['schema']}### Hint:
    {sample["hint"]}
    ### Schema_links:{sample["predicted_schema_linking"]}
    [/INST]
    Response:
    {sample['gold_query']}"""
  result += tokenizer.eos_token
  sample["text"] = result
  return sample

def get_datasets(file_training, file_eval):
    path_training = f"Prepared_Data/{file_training}"
    path_eval = f"Prepared_Data/{file_eval}"

    train_dataset = load_dataset("csv",data_files=path_training, split="train")
    eval_dataset = load_dataset("csv",data_files=path_eval, split="train")

    return train_dataset, eval_dataset

train_dataset, eval_dataset = get_datasets("train.csv", "eval.csv")

train_dataset = train_dataset.map(format_instruction_SQL_Generation, remove_columns=[f for f in train_dataset.features if not f == 'text'])
eval_dataset = eval_dataset.map(format_instruction_SQL_Generation, remove_columns=[f for f in eval_dataset.features if not f == 'text'])

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = f"Models/{new_model}/checkpoints"

# Number of training epochs
num_train_epochs = 4

# Batch size per GPU for training
per_device_train_batch_size = 2
per_device_eval_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 2

# Optimizer to use
optim = "paged_adamw_32bit"

# Evaluation after every X updates steps
eval_delay = 250 # in other notebook 1000

# Save checkpoint every X updates steps
save_steps = 250 # in other notebook 1000

# Log every X updates steps
logging_steps = 250 # in other notebook 1000

# Initial learning rate (AdamW optimizer)
learning_rate = 5e-5

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Maximum gradient normal (gradient clipping)
max_grad_norm = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

################################################################################
# SFT parameters
################################################################################

# Define a response template string that contains the prefix "### Schema_links:"
response_template = "Response:"
# Encode the response template string using the tokenizer, excluding special tokens, and get the token IDs
# The [1:] index is used to exclude the initial token, as it's not necessary for the completion-only LM
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
# Create a DataCollatorForCompletionOnlyLM object, which is used to collate data for completion-only language modeling tasks
# It takes the token IDs of the response template and the tokenizer as inputs
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# Maximum sequence length to use
max_seq_length = 7400

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Set training parameters
training_arguments = TrainingArguments(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    output_dir = output_dir,
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    evaluation_strategy = "steps",
    eval_delay = eval_delay,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    max_grad_norm = max_grad_norm,
    bf16 = bf16,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    lr_scheduler_type = lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset.shuffle(seed=42),
    eval_dataset = eval_dataset.shuffle().select(range(1500)),
    dataset_text_field = "text",
    tokenizer = tokenizer,
    data_collator = collator,
    args = training_arguments,
    max_seq_length = max_seq_length,
    packing = packing,
)

# Train model
trainer.train()
