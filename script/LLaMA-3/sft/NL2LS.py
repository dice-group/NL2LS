from transformers import BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, pipeline, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Do not use device_map='auto'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True,
    low_cpu_mem_usage=True
)
model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
model.config.pretraining_tp = 1

# Use DataParallel to wrap the model
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

def generate_answer(example):
    prompt = pipe.tokenizer.apply_chat_template(example["messages"][:2],
                                                tokenize=False,
                                                add_generation_prompt=True)
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("")
    ]
    
    outputs = pipe(prompt,
                   max_new_tokens=512,
                   eos_token_id=terminators,
                   do_sample=True,
                   temperature=0.6,
                   top_k=50,
                   top_p=0.9)
    generated_text = outputs[0]['generated_text']
    return {"ls": example['0'], "generated_text": generated_text}

def create_input_prompt(example):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": example["1"]},
            {"role": "assistant", "content": example["0"]}
        ]
    }

system_message = """
Translate text into Link Specification: 
"""

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Initialize pipeline using the underlying model
pipe = pipeline("text-generation", model=model.module if isinstance(model, torch.nn.DataParallel) else model, tokenizer=tokenizer)

# Read CSV files
train_df = pd.read_csv("../../../datasets/New-Datasets/limes-silver/train.txt", sep="\t")
val_df = pd.read_csv("../../../datasets/New-Datasets/limes-silver/dev.txt", sep="\t")

# Convert DataFrame to Dataset
train_dataset = Dataset.from_pandas(train_df.head(200))
val_dataset = Dataset.from_pandas(val_df.head(20))

train_dataset = train_dataset.map(create_input_prompt)
val_dataset = val_dataset.map(create_input_prompt)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="nl2ls_models",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    logging_steps=5,
    warmup_ratio=0.03,
    warmup_steps=100,
    group_by_length=True,
    lr_scheduler_type="constant",
    dataloader_num_workers=1,
    push_to_hub=False
)

# Ensure trainer uses the correct model
trainer = SFTTrainer(
    model=model.module if isinstance(model, torch.nn.DataParallel) else model,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512
)

trainer.train()
trainer.model.save_pretrained("nl2ls_models")
