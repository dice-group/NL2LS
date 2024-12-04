from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import re

# Function to compute cosine similarity
def compute_similarity(ls, line, model):
    similarities = []
    embeddings = model.encode([ls, line], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    similarities.append(similarity)
    return similarities
# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the file
file_path = '../new-datasets/limes-silver/train.txt'  # Replace with the correct path
with open(file_path, 'r') as file:
    data = file.readlines()

# Read the file
file_path = '../new-datasets/limes-silver/test.txt'  # Replace with the correct path
with open(file_path, 'r') as file:
    data_test = file.readlines()

f = open("test_dataset.txt", "w")
f.write("ls\tnl\tls_sample\tnl_sample\n")
for number, test in tqdm(enumerate(data_test)):
    if number == 0:
        continue
    test = test.strip()
    test_ls = test.split("\t")[0].strip()
    test_nl = test.split("\t")[1].strip()
    highest_score = 0
    ls_sample = ""
    nl_sample = ""
    for num, line in enumerate(data):
        if num==0:
            continue
        line_ls = line.split("\t")[0].strip()
        line_nl = line.split("\t")[1].strip()
        score = compute_similarity(test_ls, line_ls, model)
        if score[0] > highest_score:
            highest_score = score[0]
            ls_sample = line_ls
            nl_sample = line_nl
    f.write(f"{test_ls}\t{test_nl}\t{ls_sample}\t{nl_sample}\n")
f.close()