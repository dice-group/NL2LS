from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model and tokenizer (on CPU)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# Path to the input file
file_path = "path/to/your/test.txt"

# Read and process file
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Skip header and generate outputs
for line in lines[1:]:
    if "\t" in line:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            sentence = parts[1].strip()
            if sentence:
                prompt = f"Translate this to link specification: {sentence}"
                result = pipe(prompt, max_length=128, do_sample=False)
                print(result[0]['generated_text'])
