import csv
import json
import os

def tsv_to_json(tsv_file_path, json_file_path):
    # Initialize an empty list to store the JSON objects
    json_array = []

    # Open and read the TSV file
    with open(tsv_file_path, mode='r', encoding='utf-8') as tsv_file:
        # Create a CSV reader object using the tab delimiter
        csv_reader = csv.DictReader(tsv_file, delimiter='\t')
        
        # Iterate over each row in the TSV file
        for row in csv_reader:
            # Map the columns to "output" and "input" keys
            json_object = {
                "output": row[csv_reader.fieldnames[0]],
                "input": row[csv_reader.fieldnames[1]]
            }
            
            # Append the JSON object to the list
            json_array.append(json_object)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # Write the JSON array to a file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(json_array, json_file, indent=4)


def convert_all_datasets(datasets_root, lang):
    datasets_dir = datasets_root
    
    for root, dirs, files in os.walk(datasets_dir):
        if len(root.split(os.sep)) - len(datasets_dir.split(os.sep)) == 1:  # First level directories
            tsv_files = ['train.txt', 'test.txt', 'dev.txt']
            found_files = set(files) & set(tsv_files)
            
            if len(found_files) == 3:  # All three files are present
                for file_name in tsv_files:
                    tsv_file_path = os.path.join(root, file_name)
                    relative_root = os.path.relpath(root, datasets_dir)
                    json_file_path = os.path.join(f'data_dir/{lang}', relative_root, file_name.replace('.txt', '.json'))
                    
                    # Create the necessary directories if they don't exist
                    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
                    
                    tsv_to_json(tsv_file_path, json_file_path)


# Example usage
if __name__ == "__main__":
    convert_all_datasets('../../../datasets_EN', 'en')
    convert_all_datasets('../../../datasets_DE', 'de')