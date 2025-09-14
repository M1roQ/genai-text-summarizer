# genai-text-summarizer
# Text Summarization with BART

This project implements a text summarization tool using the `facebook/bart-large-cnn` model from Hugging Face Transformers.

## Project structure

- `res/input.txt` — example input text file
- `res/summary.txt` — example output summary file
- `requirements.txt` — list of Python dependencies
- `summarizer.py` — main script for summarization

---

## Installation

1. Clone the repository or download files.

2. Create and activate a Python virtual environment.

### On Linux/macOS:

```
python3 -m venv env
source env/bin/activate
```

### On Windows PowerShell:

```
python -m venv env
.\env\Scripts\Activate.ps1
```

3. Install required packages:

```
pip install -r requirements.txt
```

---

## Usage

Run the summarization script with input and output file paths:

```
python summarizer.py --input res/input.txt --output res/summary.txt
```

- `--input` — path to the input text file (e.g. `res/input.txt`)
- `--output` — path to save the summarized text (e.g. `res/summary.txt`)

---

## Example

Given the input file `res/input.txt` with a long text, the script will generate a concise summary and save it to `res/summary.txt`.

---


## Notes

- The model was trained on English-language texts, so it is recommended to use English input for best results. Using other languages may reduce summarization quality.
