import re

class NLtoLinkSpecification:
    def __init__(self, namespace="x"):
        self.namespace = namespace

    def natural_language_to_ls(self, natural_text):
        """
        Extended rule-based parser for natural language to LS conversion.
        Handles phrases like:
        - "overlap by at least 72%"
        - "touch with 82% confidence"
        - "city names (Levenshtein)"
        - "names (measured by Ratcliff similarity) are at least 56%"
        """
        patterns = []

        # Pattern 1: Overlap/contain/touch with threshold (e.g., "touch with 82% confidence")
        generic_matches = re.findall(r"(overlap|contain|touch)[s]? with (\d+)%", natural_text, re.IGNORECASE)
        for sim, threshold in generic_matches:
            sim_func = sim.lower()
            patterns.append(f"{sim_func}({self.namespace}.geo,{self.namespace}.geo)|{float(threshold) / 100:.2f}")

        # Pattern 2: Overlap/contain by threshold (e.g., "overlap by at least 72%")
        geo_matches = re.findall(r"(overlap|contain)[s]? by at least (\d+)%", natural_text, re.IGNORECASE)
        for sim, threshold in geo_matches:
            sim_func = sim.lower()
            patterns.append(f"{sim_func}({self.namespace}.geo,{self.namespace}.geo)|{float(threshold) / 100:.2f}")

        # Pattern 3: field names (measured by Similarity) + threshold
        field_matches = re.findall(
            r"(?:their|the)\s+([\w\s]+?)\s+\(measured by\s+(\w+)\s+similarity\)\s+are(?: at least| above)?\s+(\d+)%",
            natural_text, re.IGNORECASE
        )
        for field, sim, threshold in field_matches:
            field_clean = field.strip().lower().replace(" ", "_")
            sim_func = sim.lower()
            patterns.append(
                f"{sim_func}({self.namespace}.{field_clean},{self.namespace}.{field_clean})|{float(threshold) / 100:.2f}")

        # Pattern 4: "<sim> similarity of <field> is <threshold>%"
        alt_matches = re.findall(r"(\w+)\s+similarity.*?([\w\s]+).*?(\d+)%", natural_text, re.IGNORECASE)
        for sim, field, threshold in alt_matches:
            field_clean = field.strip().lower().replace(" ", "_")
            sim_func = sim.lower()
            patterns.append(
                f"{sim_func}({self.namespace}.{field_clean},{self.namespace}.{field_clean})|{float(threshold) / 100:.2f}")

        # Pattern 5: "<field> names (SimilarityFunc)" without threshold → assume 0.50
        approx_matches = re.findall(r"([\w\s]+?) names\s*\((\w+)\)", natural_text, re.IGNORECASE)
        for field, sim in approx_matches:
            field_clean = field.strip().lower().replace(" ", "_")
            sim_func = sim.lower()
            patterns.append(f"{sim_func}({self.namespace}.{field_clean},{self.namespace}.{field_clean})|0.50")

        if not patterns:
            return f"# ERROR: Could not parse line: {natural_text.strip()}"

        return f"AND({','.join(patterns)})"


def convert_file(input_path, output_path):
    converter = NLtoLinkSpecification()
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        lines = infile.readlines()

        # Skip header
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    natural_text = parts[1]
                    ls = converter.natural_language_to_ls(natural_text)
                    outfile.write(ls + "\n")
                else:
                    outfile.write(f"# ERROR: Line malformed: {line.strip()}\n")

# File paths
input_file = "/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/silver-silk-datasets/osl_test_dataset.txt"
output_file = "/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/script/base_line_model_rule_based/results/silver-silk-datasets_results.txt"

convert_file(input_file, output_file)
print(f"✔ Link Specifications written to {output_file}")





# file paths
# Geo-Spacial_nearby_proximity_and_temporal
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/Geo-Spacial_nearby_proximity_and_temporal/Geo-Spacial-nearby.txt
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/Geo-Spacial_nearby_proximity_and_temporal/Geo-spacial-proximity.txt
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/Geo-Spacial_nearby_proximity_and_temporal/temporal.txt

# limes-annotated
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/limes-annotated/osl_test_dataset.txt

# limes-manipulated
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/limes-manipulated/osl_test_dataset.txt

# limes-silver
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/limes-silver/osl_test_dataset.txt

# silk-annotated
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/silk-annotated/osl_test_dataset.txt

# silver-limes-manipulated
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/silver-limes-manipulated/osl_test_dataset.txt

# silver-silk-datasets
#/home/reda/Desktop/Academic work/My papers/NL2LS-ISWC_2025/datasets/silver-silk-datasets/osl_test_dataset.txt
