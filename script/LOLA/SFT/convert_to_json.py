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


# Example usage
if __name__ == "__main__":
    tsv_to_json('../../../datasets_EN/limes-silver/train.txt', 'data_dir/en/limes-silver/train.json')