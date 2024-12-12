import torch
import openai
import pandas as pd
torch.cuda.empty_cache()

# Define your OpenAI API key
openai.api_key = "open-ai-key"

def query_llm(prompt):
    """
    Send a prompt to the LLM and return the response.
    """
    print("Prompt sent to LLM:")
    print(prompt)
    response = openai.ChatCompletion.create(
        messages=prompt,
        model="gpt-4o-2024-08-06"  # Update the model as needed
    )
    return {"response": response['choices'][0]['message']['content']}

def gpt_4o_parse_to_ls(verbalized_text):
    """
    Uses GPT-4o to parse a verbalized description into a Link Specification (LS).
    :param verbalized_text: The natural language description of the LS.
    :return: The parsed LS in machine-readable format.
    """
    # Define the refined prompt for the LLM
    prompt = [
        {"role": "system", "content": "You are an expert in translating verbalized logic into machine-readable format."},
        {"role": "user", "content": f"""
        You are an expert in knowledge graphs and link specifications. Your task is to translate a natural language description of link specifications into a machine-readable format. 

        The input description is as follows:
        "{verbalized_text}"

        The output should follow these rules:
        - Use the format: <metric>(<entity1>, <entity2>) | <threshold>
        - Logical operators like "or" and "and" are binary operators with two operands separated by a comma and should be written as OR() and AND(). operators with more than 2 operands must by neasted
        - Conditions should be grouped and nested based on their logical hierarchy in the input description.
        - Do not flatten nested logical structures.
        - The final output should reflect the exact logical grouping implied in the input.

        Translate the given description into the desired format with the correct nested structure.
        Provide the answer only in one line.
        """}
    ]

    # Query the LLM using the query_llm function
    response = query_llm(prompt)

    # Extract and return the result
    return response["response"]

df = pd.read_csv(f"../../new-datasets/limes-annotated/test.txt", sep="\t")
test_list = df['1'].tolist()
ls_list = df['0'].tolist()
f = open(f"results.txt", "w")
f_ls = open(f"ls.txt", "w")
for num, nl in enumerate(test_list):
    print(f"generating ls {num+1}/{len(test_list)}")
    verbalized_text = f"""{nl}"""
    ls = gpt_4o_parse_to_ls(verbalized_text)
    f.write(f"{ls}\n")
    f_ls.write(f"{ls_list[num]}\n")

f.close()
f_ls.close()