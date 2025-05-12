# Sample usage: python inference_example.py
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

model_id = "dice-research/lola_v1"
lora_dir = "./trained_model/ds-lola_v1-en-limes-silver"

input_template = "Translate text into Link Specification:\n{}\n"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    trust_remote_code=True,
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained(
    lora_dir,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

base_model.resize_token_embeddings(len(tokenizer))

base_model.load_adapter(lora_dir)
model_instance = base_model.to('cuda')

# Read input from TSV file
input_file = "../../../datasets_EN/limes-silver/test.txt"  # Replace with your TSV file path
df = pd.read_csv(input_file, sep='\t', header=0)
sample_texts = df.iloc[:, 1].tolist()  # Extract texts from the second column

# Introduce batch size of 8
batch_size = 64
batches = [sample_texts[i:i + batch_size] for i in range(0, len(sample_texts), batch_size)]

generated_texts = []

for batch in tqdm(batches, desc='Processing batches'):
    formatted_batch = [input_template.format(text) for text in batch]
    
    model_inputs = tokenizer(
        formatted_batch,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    ).to('cuda')

    output_sequences = model_instance.generate(**model_inputs, eos_token_id=tokenizer.eos_token_id, max_length=tokenizer.model_max_length)
    
    generated_texts.extend(tokenizer.batch_decode(output_sequences, skip_special_tokens=False))

# Create the output directory
output_path = "./output"
input_path_parts = input_file.split(os.sep)[-3:]
for part in input_path_parts:
    output_path = os.path.join(output_path, part)

os.makedirs(output_path, exist_ok=True)

# Write the output to a TSV file
with open(output_path, 'w', encoding='utf-8') as f:
    for text in generated_texts:
        f.write(text + '\n')

print(f"Output written to {output_path}")