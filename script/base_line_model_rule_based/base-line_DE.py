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
input_file = "path/to/your/test.txt"
output_file = "path/to/your/results.txt"

convert_file_de(input_file, output_file)
print(f"✔ Link-Spezifikationen gespeichert in {output_file}")

