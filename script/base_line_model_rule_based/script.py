import re

class NLtoLinkSpecification:
    def __init__(self, namespace="uri"):
        self.namespace = namespace

    def natural_language_to_ls(self, natural_text):
        """
        Converts a single natural language sentence to a Link Specification.
        Example: "Match 'Has Name' with 'Full Name' using Trigram similarity."
        Returns: "trigram(uri:has_name,uri:full_name)"
        """
        match = re.search(r"Match '(.*?)' with '(.*?)' using (\w+)", natural_text, re.IGNORECASE)
        if not match:
            return f"# ERROR: Could not parse line: {natural_text.strip()}"  # Return error as comment

        left_property, right_property, measure = match.groups()
        left_property = self.clean_property(left_property)
        right_property = self.clean_property(right_property)
        return f"{measure.lower()}({self.namespace}:{left_property},{self.namespace}:{right_property})"

    def clean_property(self, prop):
        return prop.lower().replace(" ", "_")


"""
def convert_file(input_path, output_path):
    converter = NLtoLinkSpecification()
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if line.strip():
                ls = converter.natural_language_to_ls(line.strip())
                outfile.write(ls + "\ n")
"""

def convert_file(input_path, output_path):
    converter = NLtoLinkSpecification()
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    natural_text = parts[1]  # Only the second column (NL description)
                    ls = converter.natural_language_to_ls(natural_text)
                    outfile.write(ls + "\n")
                else:
                    outfile.write(f"# ERROR: Line malformed: {line.strip()}\n")



# Example usage:
input_file = "natural_language_input.txt"     # Input: one NL sentence per line
output_file = "link_specifications_output.txt"  # Output: one LS per line

convert_file(input_file, output_file)
print(f"✔ Link Specifications written to {output_file}")












