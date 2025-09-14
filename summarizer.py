import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sys

def read_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError("File is empty")
        return text
    except Exception as e:
        raise IOError(f"Error reading file '{filepath}': {e}")

def write_text_file(filepath, text):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise IOError(f"Error writing to file '{filepath}': {e}")

def summarize_text(text, model_name="facebook/bart-large-cnn", max_length=50, min_length=25):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Text summarization with BART")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to output summary file")
    args = parser.parse_args()

    try:
        text = read_text_file(args.input)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        summary = summarize_text(text)
    except Exception as e:
        print(f"Error during summarization: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        write_text_file(args.output, summary)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(3)

    print(f"Summary successfully saved to '{args.output}'")

if __name__ == "__main__":
    main()
