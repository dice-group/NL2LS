# Sample usage: python inference_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


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

sample_texts = ["a link will be generated if the givenName of the source and the streetName of the target have a Cosine similarity of 45% or a Qgrams similarity of 25% or the streetNames of the source and the target have a Jarowinkler similarity of 45% and the givenName of the source and the streetName of the target have a Ratcliff similarity of 25% and the streetNames of the source and the target have a Qgrams similarity of 25%", "a link will be generated if the givenName of the source and the streetName of the target have a Jaccard similarity of 0% or a Trigrams similarity of 100% or the streetNames of the source and the target have a Mongeelkan similarity of 0% or the givenName of the source and the streetName of the target have a Soundex similarity of 100% or the streetNames of the source and the target have a Trigrams similarity of 100%"]

sample_texts = [input_template.format(text) for text in sample_texts]

model_inputs = tokenizer(
    sample_texts,
    return_tensors="pt",
    padding="longest",
    truncation=True,
).to('cuda')

output_sequences = model_instance.generate(**model_inputs, eos_token_id=tokenizer.eos_token_id, max_length=tokenizer.model_max_length)

generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=False)
for text in generated_texts:
    print(text)