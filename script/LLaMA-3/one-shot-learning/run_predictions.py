from transformers import BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, pipeline
import pandas as pd
from datasets import Dataset 
import torch
from transformers import DataCollatorForSeq2Seq

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device_map = {"": 0}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(  ## If it fails at this line, restart the runtime and try again.
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True,
    low_cpu_mem_usage=True
)
model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
model.config.pretraining_tp = 1

def generate_answer(example):
    
    prompt = pipe.tokenizer.apply_chat_template(example["messages"][:2],
                                                tokenize=False,
                                                add_generation_prompt=True)
    terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
    
    outputs = pipe(prompt,
                max_new_tokens=100,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.3
                )
    generated_text = outputs[0]['generated_text'][len(prompt):]
    return {"ls": example['ls'], "generated_text": generated_text}

def create_input_prompt(example):
    system_message = f"""
    Translate text into Link Specification:
    example text:
    {example["nl_sample"]}

    example link specifications:
    {example["ls_sample"]}

    Provide the answer only in one line and no need for further explanation.
    """
    print(system_message)
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": example["nl"]},
        ]
    }

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
    
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Read CSV files
test_df = pd.read_csv(f"../../../new-datasets/limes-manipulated/osl_test_dataset.txt", sep="\t")
# Convert DataFrame to Dataset
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(create_input_prompt)

# Generate answers for the dataset
results = test_dataset.map(generate_answer, batched=False)

f = open(f"predicts.txt", "w")
f_ls = open(f"ls.txt", "w")
for result in results:
    #print("#############")
    #print(result["generated_text"])
    #print("########################")
    output = result["generated_text"]
    f.write(f"{output}\n")
    f_ls.write(f"{result['ls']}\n")
f.close()
f_ls.close()