# AI Benchmark Runner

This project is a small benchmark runner for testing how different AI models answer graph and chart questions. It loads questions from a CSV file, matches them with graph assets, sends them to multiple model providers, records the answers, tracks latency, and saves the results for review.

It is meant to be practical rather than flashy: you give it questions, point it at some graph files, add API keys, and it handles the rest.

## What it does

The runner can:
- load questions from a CSV file
- find matching graph images and graph data files
- send each question to several AI models
- collect responses and errors
- track how long each model takes
- save outputs as CSV, JSON, and log files
- keep track of progress so a run can resume later

## Main folders

- [src](src) – the Python logic for the runner
  - [src/runner.py](src/runner.py) – the main script
  - [src/models.py](src/models.py) – model list and provider settings
  - [src/questions.py](src/questions.py) – question loading and graph matching
  - [src/batch_scheduler.py](src/batch_scheduler.py) – progress tracking for batches
  - [src/grade_results.py](src/grade_results.py) – grading helper
- [data](data) – question files
- [assets](assets) – graph assets
- [results](results) – output files from each run
- [.env](.env) – API keys and secrets
- [requirements.txt](requirements.txt) – Python dependencies

## Requirements

This project is built with Python and a handful of libraries for model access, image handling, and data output.

### Python
- Python 3.11.x is the most reliable choice

### Main packages
- python-dotenv
- google-genai
- groq
- huggingface-hub
- openai
- Pillow
- pandas
- matplotlib
- seaborn
- scipy
- pytest

## Getting started

### 1. Install dependencies

From the project root, run:

```bash
python3 -m pip install -r requirements.txt
```

### 2. Add your API keys

Open [.env](.env) and fill in the values you need.

The runner expects these variables:
- GOOGLE_API_KEY
- GROQ_API_KEY
- HF_TOKEN
- NVIDIA_API_KEY
- OPENROUTER_API_KEY
- HCOMPANY_API_KEY

### 3. Add your questions

Put your question list in [data/questions.csv](data/questions.csv).

A typical CSV should include columns such as:
- Q#
- Graph
- Question Type
- Question
- Answer

### 4. Add graph files

For each question, the program looks for the matching graph image and graph CSV data.

A simple setup is to place files under:
- [assets/graphs](assets/graphs)
- [Graphs](Graphs)

The graph name in the CSV should match the files you provide as closely as possible.

### 5. Turn models on or off

Open [src/models.py](src/models.py) and adjust the `enabled` value for each model.
- `True` means the model will be used
- `False` means it will be skipped

### 6. Adjust runner settings if needed

Open [src/runner.py](src/runner.py) and look near the top of the file.

You can tweak:
- `QUESTIONS_PER_DAY`
- `INTER_QUESTION_DELAY`
- `MODEL_TIMEOUT`
- `RETRY_DELAY_S`
- `PROVIDER_CONCURRENCY`

The defaults are fine for a first run.

### 7. Run the benchmark

From the project root, run:

```bash
python3 -m src.runner
```

If that does not work, try:

```bash
python3 src/runner.py
```

## What to expect while it runs

When the run starts, you should see:
- the run directory
- how many models are being used
- how many questions are being processed
- progress for each question

You will also see:
- `✅` for successful responses
- `❌` for errors or failed calls

At the end, the script prints a summary with the total number of successful responses and the number of errors.

## Output files

Each run writes files into [results](results) inside a timestamp-based folder.

You should usually see:
- `questions_results.csv`
- `combined.json`
- `run_log.txt`

These files contain the benchmark answers, model metadata, latency, and any issues that came up.

## Logs and debugging

The main log file is:
- [results](results)/run_log.txt

That file is useful when a model fails or you want to understand what happened during the run.

## Batch state and restarting

The runner keeps track of completed questions in:
- [results](results)/batch_state.json

That file is helpful if you want to resume a run later.

If you want to start over from scratch, delete the batch state file and run again.

## Grading results

Once the run is complete, you can use [src/grade_results.py](src/grade_results.py) to grade the outputs and compare them against the expected answers.

## Notes

This project is still fairly hands-on, so the main things you need to manage are:
- your API keys
- your question CSV
- your graph files
- which models you want enabled

That said, once those are in place, the runner is pretty straightforward to use.
