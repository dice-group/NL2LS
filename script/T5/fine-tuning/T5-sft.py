import torch
from transformers import pipeline

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

# Set model name
base_model_name = "t5-small" # t5-large

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load training and validation data
train_path = "/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/datasets_DE/limes-silver/train.txt"
val_path = "/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/datasets_DE/limes-silver/dev.txt"

train_df = pd.read_csv(train_path, sep="\t").dropna()
val_df = pd.read_csv(val_path, sep="\t").dropna()
train_df.columns = ["target", "source"]
val_df.columns = ["target", "source"]

# Task prefix
task_prefix = "Translate text into Link Specification: "

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenization with label padding fix
def preprocess_function(examples):
    inputs = [task_prefix + src for src in examples["source"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(examples["target"], max_length=256, truncation=True, padding="max_length")
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="nl2ls_t5_models",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    #evaluation_strategy="epoch",
    remove_unused_columns=True,
    push_to_hub=False
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("nl2ls_t5_models")
tokenizer.save_pretrained("nl2ls_t5_models")

# Evaluate and plot training loss
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# Plotting loss from logs (if enabled in training_args)
log_history = trainer.state.log_history
train_loss = [x["loss"] for x in log_history if "loss" in x]
steps = list(range(1, len(train_loss) + 1))

# Reload fine-tuned model and tokenizer
model_path = "nl2ls_t5_models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

# Load test data
test_path = "/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/datasets_DE/limes-silver/test.txt"
test_df = pd.read_csv(test_path, sep="\t").dropna()
test_df.columns = ["target", "source"]

# Convert to HF dataset
test_dataset = Dataset.from_pandas(test_df)

# Generation function
def generate_answer(example):
    input_text = task_prefix + example["source"]
    output = pipe(input_text, max_new_tokens=100, do_sample=False)
    return {
        "target": example["target"],
        "generated_text": output[0]["generated_text"]
    }

# Generate predictions
results = test_dataset.map(generate_answer)

# Save results
with open("results_t5.txt", "w") as f_out, open("ls_t5.txt", "w") as f_ls:
    for r in results:
        f_out.write(f"{r['generated_text']}\n")
        f_ls.write(f"{r['target']}\n")