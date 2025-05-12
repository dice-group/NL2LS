# Sample usage: python tsv_batch_inference.py
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

model_id = "dice-research/lola_v1"
lora_dir = "./trained_model/ds-lola_v1-en-limes-silver"
input_file = "../../../datasets_EN/limes-silver/test.txt"
output_root = "./output"

batch_size = 64

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
df = pd.read_csv(input_file, sep='\t', header=0)
sample_texts = df.iloc[:, 1].tolist()  # Extract texts from the second column


batches = [sample_texts[i:i + batch_size] for i in range(0, len(sample_texts), batch_size)]

output_path = output_root
input_path_parts = input_file.split(os.sep)[-3:]
file_name = input_path_parts[-1]
for part in input_path_parts[:-1]:
    output_path = os.path.join(output_path, part)

os.makedirs(output_path, exist_ok=True)

print(f'Output directory: {output_path}')

output_path = os.path.join(output_path, file_name)

# Open the output file for writing
with open(output_path, 'w', encoding='utf-8') as f:
    for batch in tqdm(batches, desc='Processing batches'):
        formatted_batch = [input_template.format(text) for text in batch]
        
        model_inputs = tokenizer(
            formatted_batch,
            return_tensors="pt",
            padding="longest", # do not change this, it will affect the output generation logic
            truncation=True,
        ).to('cuda')
        
        seq_len = len(model_inputs['input_ids'][0]) # all sequence lengths are the same due to padding
        
        output_sequences = model_instance.generate(**model_inputs, eos_token_id=tokenizer.eos_token_id, max_length=tokenizer.model_max_length)
        
        # Cutting out the input and the prompt
        output_sequences = [seq[seq_len:] for seq in output_sequences]
        
        generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        # Write the output for this batch to the file
        for text in generated_texts:
            f.write(text + '\n')

print(f"Output written to {output_path}")