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
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_k=50,
                top_p=0.9,
                )
    generated_text = outputs[0]['generated_text']
    return {"ls": example['0'], "generated_text": generated_text}

def create_input_prompt(example):
    return {
        "messages": [
            {"role": "system","content": system_message},
            {"role": "user", "content": example["1"]},
        ]
    }

system_message = """
    Translate text into Link Specification:
    a link will be generated if the givenName of the source and the streetName of the target have a Cosine similarity of 45% or a Qgrams similarity of 25% or the streetNames of the source and the target have a Jarowinkler similarity of 45% and the givenName of the source and the streetName of the target have a Ratcliff similarity of 25% and the streetNames of the source and the target have a Qgrams similarity of 25%
    
    The output is: 
    AND(OR(cosine(x.givenName,y.streetName)|0.45,AND(OR(qgrams(x.givenName,y.streetName)|0.25,jaroWinkler(x.streetName,y.streetName)|0.45)|0.45,ratcliff(x.givenName,y.streetName)|0.25)|0.45)|0.45,qgrams(x.streetName,y.streetName)|0.25)

    No explanations are needed.
    """

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
    
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Read CSV files
test_df = pd.read_csv(f"datasets/few-shot-learning-dataset/limes-silver/test.txt", sep="\t")
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
    output = result["generated_text"].split("\n")[-1]
    f.write(f"{output}\n")
    f_ls.write(f"{result['ls']}\n")
f.close()
f_ls.close()