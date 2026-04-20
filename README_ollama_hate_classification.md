## Local Ollama Hate Classification

This workspace now includes `classify_hate_ollama.py`, a standalone script for classifying Telugu or Telugu-mixed comments into:

- `0` = non-hate
- `1` = hate

### What the script does

- Reads a single `.csv` or `.xlsx` file, or processes an entire folder of `.csv` and `.xlsx` files.
- Detects the text column automatically, or you can pass `--text-column`.
- Normalizes the text before inference.
- Converts emojis into short English expressions such as `<emoji: fire>` or `<emoji: face with tears of joy>`.
- Replaces URLs and mentions with placeholders so the model sees cleaner text.
- Groups comments into batches based on total normalized text size.
- Sends a strict JSON-only prompt to the local Ollama HTTP API.
- Validates the returned JSON and writes a new output CSV.

### Install requirements

```powershell
& 'C:\Users\HP\anaconda3\python.exe' -m pip install -r .\requirements.txt
```

### Single-file example

Start Ollama separately, then run:

```powershell
& 'C:\Users\HP\anaconda3\python.exe' .\classify_hate_ollama.py `
  --input .\Politics.csv `
  --model llama3.1:8b `
  --output .\Politics_classified.csv
```

### Folder example

This processes every `.csv` and `.xlsx` file inside the folder and writes outputs into `classified_outputs` by default:

```powershell
& 'C:\Users\HP\anaconda3\python.exe' .\classify_hate_ollama.py `
  --input . `
  --model llama3.1:8b
```

To scan subfolders too:

```powershell
& 'C:\Users\HP\anaconda3\python.exe' .\classify_hate_ollama.py `
  --input . `
  --model llama3.1:8b `
  --recursive
```

For a quick smoke test:

```powershell
& 'C:\Users\HP\anaconda3\python.exe' .\classify_hate_ollama.py `
  --input .\Politics.csv `
  --model llama3.1:8b `
  --limit 25 `
  --output .\Politics_sample_classified.csv
```

### Useful options

- `--text-column Comment`
- `--ollama-url http://127.0.0.1:11434/api/generate`
- `--recursive`
- `--batch-char-budget 2200`
- `--max-batch-size 20`
- `--temperature 0`
- `--timeout 180`

### Output columns

The output file keeps the original columns and adds:

- `normalized_text`
- `hate_label`
- `ollama_raw_response`

### Notes

- The script expects Ollama to be reachable on the local HTTP API.
- In this environment, `http://127.0.0.1:11434` was not reachable during setup, so the script is ready but could not be executed end-to-end here.
- If your model sometimes returns malformed JSON, reduce `--max-batch-size` or `--batch-char-budget`.
