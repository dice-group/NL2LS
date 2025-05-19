import re

class NLtoLinkSpecificationGerman:
    def __init__(self, namespace="x"):
        self.namespace = namespace

    def _normalize_field(self, field):
        replacements = {
            "titel": "title",
            "namen": "name",
            "ereignisse": "event",
            "ereignis": "event",
            "jahr": "date",
            "straße": "streetName",
            "straßenname": "streetName",
            "name": "name"
        }
        field = field.lower()
        return replacements.get(field, field)

    def _normalize_similarity(self, sim):
        sim = sim.lower()
        return {
            "q-grams": "qgrams",
            "qgrams": "qgrams",
            "cosine": "cosine",
            "jaro-winkler": "jaroWinkler",
            "levenshtein": "levenshtein",
            "overlap": "overlap",
            "ratcliff": "ratcliff",
        }.get(sim, sim)

    def natural_language_to_ls(self, text):
        """
        Parses German NL sentence into a LIMES-style LS expression.
        Handles patterns with thresholds, logical ops, and similarity functions.
        """
        conditions = []

        # Match patterns like: Q-grams ≥ 0.54 or Cosine ≥ 0.41
        sim_patterns = re.findall(r"(\w+(?:-\w+)?)\s*(?:≥|>=|=|ist|beträgt)\s*([0-9]+(?:[.,][0-9]+)?)", text, re.IGNORECASE)
        for sim_func, threshold in sim_patterns:
            sim_func = self._normalize_similarity(sim_func)
            try:
                value = float(threshold.replace(",", "."))
            except ValueError:
                continue
            # Apply default field placeholder
            conditions.append(f"{sim_func}({self.namespace}.feld,{self.namespace}.feld)|{value:.2f}")

        # Logical structure replacement (simple handling for now)
        logic_expr = text.lower()
        logic_expr = logic_expr.replace(" und ", " AND ")
        logic_expr = logic_expr.replace(" oder ", " OR ")

        # Combine into LS
        if not conditions:
            return f"# ERROR: Konnte Zeile nicht interpretieren: {text.strip()}"
        
        operator = "AND" if "AND" in logic_expr and "OR" not in logic_expr else "OR" if "OR" in logic_expr else "AND"
        return f"{operator}({','.join(conditions)})"


def convert_file_de(input_path, output_path):
    converter = NLtoLinkSpecificationGerman()
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        lines = infile.readlines()

        for line in lines[1:]:  # skip header
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    natural_text = parts[1]
                    ls = converter.natural_language_to_ls(natural_text)
                    outfile.write(ls + "\n")
                else:
                    outfile.write(f"# ERROR: Fehlerhafte Zeile: {line.strip()}\n")

# Example usage
#RTC:NL2LS-ISWC_2025/datasets_DE/limes-manipulated/test.txt
#limes-silver
#silk-annotated
#silver-limes-manipulated
#silver-silk-datasets
input_file = "/local/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/datasets_DE/silver-silk-datasets/test.txt"
output_file = "/local/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/script/base-line/results_DE/results.txt"

convert_file_de(input_file, output_file)
print(f"✔ Link-Spezifikationen gespeichert in {output_file}")






# import re

# class NLtoLinkSpecification:
#     def __init__(self):
#         pass

#     def natural_language_to_ls_en(self, natural_text):
#         """
#         Rule-based parser for English sentences to LS.
#         """
#         patterns = []

#         # Pattern 1: "overlap by at least 72%"
#         generic_matches = re.findall(r"(overlap|contain|touch)[s]? with (\d+)%", natural_text, re.IGNORECASE)
#         for sim, threshold in generic_matches:
#             sim_func = sim.lower()
#             patterns.append(f"{sim_func}(x.geo,y.geo)|{float(threshold) / 100:.2f}")

#         # Pattern 2: "overlap by at least 72%"
#         geo_matches = re.findall(r"(overlap|contain)[s]? by at least (\d+)%", natural_text, re.IGNORECASE)
#         for sim, threshold in geo_matches:
#             sim_func = sim.lower()
#             patterns.append(f"{sim_func}(x.geo,y.geo)|{float(threshold) / 100:.2f}")

#         # Pattern 3: "field (measured by X similarity) are at least 56%"
#         field_matches = re.findall(
#             r"(?:their|the)\s+([\w\s]+?)\s+\(measured by\s+(\w+)\s+similarity\)\s+are(?: at least| above)?\s+(\d+)%",
#             natural_text, re.IGNORECASE
#         )
#         for field, sim, threshold in field_matches:
#             field_clean = field.strip().lower().replace(" ", "_")
#             sim_func = sim.lower()
#             patterns.append(f"{sim_func}(x.{field_clean},y.{field_clean})|{float(threshold) / 100:.2f}")

#         # Pattern 4: "X similarity of Y is Z%"
#         alt_matches = re.findall(r"(\w+)\s+similarity.*?([\w\s]+).*?(\d+)%", natural_text, re.IGNORECASE)
#         for sim, field, threshold in alt_matches:
#             field_clean = field.strip().lower().replace(" ", "_")
#             sim_func = sim.lower()
#             patterns.append(f"{sim_func}(x.{field_clean},y.{field_clean})|{float(threshold) / 100:.2f}")

#         # Pattern 5: "names (SimilarityFunc)" without threshold
#         approx_matches = re.findall(r"([\w\s]+?) names\s*\((\w+)\)", natural_text, re.IGNORECASE)
#         for field, sim in approx_matches:
#             field_clean = field.strip().lower().replace(" ", "_")
#             sim_func = sim.lower()
#             patterns.append(f"{sim_func}(x.{field_clean},y.{field_clean})|0.50")

#         if not patterns:
#             return f"# ERROR: Could not parse line: {natural_text.strip()}"

#         return f"AND({','.join(patterns)})"

#     def natural_language_to_ls_de(self, natural_text):
#         """
#         Rule-based parser for German sentences to LS.
#         """
#         patterns = []

#         # Pattern: "Overlap-Ähnlichkeit von 62 %"
#         sim_matches = re.findall(r"(\w+)-?Ähnlichkeit von\s*(\d+)[ %]*", natural_text, re.IGNORECASE)
#         for sim, threshold in sim_matches:
#             sim_func = sim.lower()
#             patterns.append(f"{sim_func}(x.field,y.field)|{float(threshold)/100:.2f}")

#         if not patterns:
#             return f"# ERROR: Could not parse line: {natural_text.strip()}"

#         return f"OR({','.join(patterns)})"

#     def natural_language_to_ls_auto(self, natural_text):
#         """
#         Auto-detect language and parse accordingly.
#         """
#         if "Ähnlichkeit" in natural_text or "Quelle" in natural_text:
#             return self.natural_language_to_ls_de(natural_text)
#         else:
#             return self.natural_language_to_ls_en(natural_text)


# def convert_file(input_path, output_path):
#     converter = NLtoLinkSpecification()
#     with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
#         lines = infile.readlines()

#         # Skip header
#         for line in lines[1:]:
#             if line.strip():
#                 parts = line.strip().split("\t")
#                 if len(parts) >= 2:
#                     natural_text = parts[1]
#                     ls = converter.natural_language_to_ls_auto(natural_text)
#                     outfile.write(ls + "\n")
#                 else:
#                     outfile.write(f"# ERROR: Line malformed: {line.strip()}\n")

#     print(f"✔ Link Specifications written to {output_path}")


# # Example file paths (edit these as needed)
# #/local/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/datasets_DE/Geo-Spacial_nearby_proximity_and_temporal
# #/local/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/datasets_DE/Geo-Spacial_nearby_proximity_and_temporal/Geo-Spacial_nearby_proximity_and_temporal_DE_test.txt

# input_file = "/local/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/datasets_DE/Geo-Spacial_nearby_proximity_and_temporal/Geo-Spacial_nearby_proximity_and_temporal_DE_test.txt"
# #/local/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/script/base-line
# output_file = "/local/upb/users/r/reih/profiles/unix/cs/NL2LS-ISWC_2025/script/base-line/results_DE/Geo-Spacial_DE_results.txt"

# convert_file(input_file, output_file)

























