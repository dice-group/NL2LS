from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import torch

# Load the T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Read input data from test.txt
input_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/datasets/Source-Datasets/limes-silver-manipulated/osl_test_dataset.txt"
#input_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/datasets/Source-Datasets/silk-silver/osl_test_dataset.txt"

#input_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/new-datasets/limes-annotated/osl_test_dataset.txt"
#input_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/new-datasets/limes-silver/osl_test_dataset.txt"
#input_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/new-datasets/limes-silver-manipulated/osl_test_dataset.txt"
#input_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/new-datasets/silk-silver/osl_test_dataset.txt"



output_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes_silver_manipulated_results.txt"
#output_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/silk_silver_results.txt"

#output_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes_annotated_results.txt"
#output_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes_silver_results.txt"
#output_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes_silver_manipulated_results.txt"
#output_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/silk_silver_results.txt"



actual_ls_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes-silver-manipulated_ls.txt"
#actual_ls_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/silk_silver_ls.txt"

#actual_ls_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes_annotated_ls.txt"
#actual_ls_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes_silver_ls.txt"
#actual_ls_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/limes_silver_manipulated_ls.txt"
#actual_ls_file = "/upb/users/r/reih/profiles/unix/cs/NL2LS/script/results/one-shot-learning/T5/silk_silver_ls.txt"

#input_file = "limes-annotated/osl_test_dataset.txt"
#output_file = "output.txt"

with open(input_file, "r") as file:
    inputs = file.readlines()

# Prepare and process inputs
results = []
lss = []
for num, line in enumerate(tqdm(inputs)):
#for num, line in enumerate(tqdm(inputs, desc="Processing Inputs")):
    if num == 0:
        continue
    line = line.strip()
    ls = line.split("\t")[0]
    nl = line.split("\t")[1]
    #if num==1:
    ls_sample = line.split("\t")[2]
    nl_sample = line.split("\t")[3]
    if not line:
        continue  # Skip empty lines
    # Append one-shot example to the input
    prompt = (
        f"Translate text into Link Specification (answering within one line):\n"
        f"Example:\nInput sentence: {nl_sample}\n"
        f"Output link spcification: {ls_sample}\n"
        f"Now solve this:\nInput: {nl}\nOutput link specification:"
    )

    #print(prompt)
    # Encode the input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate the output
    outputs = model.generate(
        input_ids,
        max_length=1024,  # Adjust as needed
        num_beams=5,  # Beam search for quality
        early_stopping=True
    )

    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append(f"{decoded_output}\n")
    lss.append(f"{ls}\n")

# Save results to output.txt
with open(output_file, "w") as file:
    file.writelines(results)

# Save results to output.txt
# actual_ls_file -- "lss.txt"

with open(actual_ls_file, "w") as file1:
    file1.writelines(lss)
print(f"Results saved to {output_file}")
