from transformers import BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, pipeline
import pandas as pd
from datasets import Dataset 
import torch
from transformers import DataCollatorForSeq2Seq
torch.cuda.empty_cache()

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

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
    return {"ls": example['0'], "generated_text": generated_text}

def create_input_prompt(example):
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": f"""
        You are an expert in knowledge graphs and link specifications. Your task is to translate a natural language description of link specifications into a machine-readable format. 

        The input description is as follows:
        "{example['1']}"

        The output should follow these rules:
        - Use the format: <metric>(<entity1>, <entity2>) | <threshold>
        - Logical operators like "or" and "and" are binary operators with two operands separated by a comma and should be written as OR() and AND(). operators with more than 2 operands must by neasted
        - Conditions should be grouped and nested based on their logical hierarchy in the input description.
        - Do not flatten nested logical structures.
        - The final output should reflect the exact logical grouping implied in the input.

        Translate the given description into the desired format with the correct nested structure.
        """},
        ]
    }

system_message = """
    You are an expert in translating verbalized logic into machine-readable format.
    Provide the answer only in one line and no need for further explanation.
    """

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
    
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Read CSV files
test_df = pd.read_csv(f"../../../new-datasets/silk-annotated/test.txt", sep="\t")
# Convert DataFrame to Dataset
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.map(create_input_prompt)

# Generate answers for the dataset
results = test_dataset.map(generate_answer, batched=False)

f = open(f"results.txt", "w")
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