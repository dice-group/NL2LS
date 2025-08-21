from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from datasets import Dataset
import pandas as pd
import argparse
import torch
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="One-shot LS generation with LOLA")
    ap.add_argument(
        "--input",
        type=str,
        default="test.csv",
        help="../../../new-datasets/limes-manipulated/osl_test_dataset.txt",
    )
    ap.add_argument(
        "--sep",
        type=str,
        default=",",
        help="Field separator for the input file (default: ','). Use '\\t' for TSV.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Max new tokens to generate per example.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="dice-research/lola_v1",
        help="LOLA model repo id.",
    )
    return ap.parse_args()


def build_one_shot_prompt(example):
    prompt = (
        "Translate the text into a Link Specification using ONE example.\n\n"
        "Example:\n"
        f"Text:\n{example['nl_sample']}\n"
        f"Link Specification:\n{example['ls_sample']}\n\n"
        "Instruction: For the user text below, output ONLY the Link Specification "
        "(single line), no explanations.\n\n"
        f"User text:\n{example['nl']}\n"
        "Answer:"
    )
    return {"prompt": prompt, "ls": example["ls"]}


def main():
    args = parse_args()

    # --- Load data
    inp_path = Path(args.input)
    if not inp_path.exists():
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    # Auto-detect TSV if user passed "\t"
    sep = "\t" if args.sep == "\\t" else args.sep
    df = pd.read_csv(inp_path, sep=sep)
    required_cols = {"ls", "nl", "ls_sample", "nl_sample"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {sorted(missing)}")

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.map(build_one_shot_prompt)

    # --- Model / Tokenizer 
    base_model_name = args.model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        pad_token_id=tokenizer.eos_token_id,
    )

    # --- Generation function
    def generate_answer(example):
        prompt = example["prompt"]
        outputs = pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            eos_token_id=[tokenizer.eos_token_id],
            return_full_text=True,
        )
        # Cut off the echoed prompt
        generated_text = outputs[0]["generated_text"][len(prompt):]
        generated_text = generated_text.strip().splitlines()[0] if generated_text.strip() else ""
        return {"ls": example["ls"], "generated_text": generated_text}

    # --- Run and write results
    results = ds.map(generate_answer, batched=False)

    with open("predicts.txt", "w", encoding="utf-8") as f_pred, \
         open("ls.txt", "w", encoding="utf-8") as f_ls:
        for r in results:
            f_pred.write(f"{r['generated_text']}\n")
            f_ls.write(f"{r['ls']}\n")

    print("Done. Wrote: predicts.txt and ls.txt")


if __name__ == "__main__":
    main()

