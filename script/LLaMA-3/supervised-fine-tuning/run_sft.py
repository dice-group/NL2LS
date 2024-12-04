from transformers import BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, pipeline, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Bits and bytes configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Load the model
device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True,
    low_cpu_mem_usage=True,
    device_map = device_map
)
model.config.use_cache = False
#model.config.pretraining_tp = 1

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the text generation pipeline
pipe = pipeline("text-generation", model=model.module if isinstance(model, torch.nn.DataParallel) else model, tokenizer=tokenizer)

def generate_answer(example):
    # Generate the input prompt for the model
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
    generated_text = outputs[0]['generated_text'][len(prompt):]
    return {"ls": example['0'], "generated_text": generated_text}

def create_input_prompt(example):
    user_message = example.get("1", "").strip()
    assistant_message = example.get("0", "").strip()

    if not user_message or not assistant_message:
        raise ValueError(f"Invalid input detected: user_message='{user_message}', assistant_message='{assistant_message}'")

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }



system_message = """
Translate text into Link Specification: 
"""

# Load the training and validation datasets
train_df = pd.read_csv("../../../new-datasets/silver-silk-datasets/train.txt", sep="\t")
val_df = pd.read_csv("../../../new-datasets/silver-silk-datasets/validation.txt", sep="\t")
print(train_df["1"])
# Remove invalid rows
train_df.dropna(inplace=True)
val_df.dropna(inplace=True)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Map to input prompt
train_dataset = train_dataset.map(create_input_prompt)
val_dataset = val_dataset.map(create_input_prompt)
# Debugging: Check the first entry of train_dataset
example_input = train_dataset[0]
example_input_str = " ".join([msg["content"] for msg in example_input["messages"]])
print("Debugging Example Input String:", example_input_str)

# Validate tokenization
tokenized_input = tokenizer(example_input_str)
print("Debugging Tokenized Input:", tokenized_input)

# PEFT Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="nl2ls_models",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust batch size
    gradient_accumulation_steps=2,  # Reduce accumulation if needed
    save_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,  # Use mixed precision if GPUs support it
    logging_steps=5,
    warmup_ratio=0.03,
    dataloader_num_workers=1,
    push_to_hub=False
)
def validate_batch(batch):
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor) and tensor.numel() == 0:
            raise ValueError(f"Empty tensor detected in batch for key: {key}")

# Modify the training loop to validate batches
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=training_args
)

# Override compute_loss to validate inputs
original_compute_loss = trainer.compute_loss
def compute_loss_with_validation(model, inputs, **kwargs):
    validate_batch(inputs)
    return original_compute_loss(model, inputs, **kwargs)

trainer.compute_loss = compute_loss_with_validation
trainer.train()

trainer.model.save_pretrained("nl2ls_models")
