import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from src.models import get_active_models
from src.questions import get_questions
from src.batch_scheduler import load_state, save_state, select_batch_questions

# Basic settings for how much work to do in one run.
QUESTIONS_PER_DAY = 180

# Small pause between questions so the providers are not hit too hard.
INTER_QUESTION_DELAY = 10

# How long each model call gets before it times out.
MODEL_TIMEOUT = 120
RETRY_DELAY_S = 1.0

# Keep things simple and avoid hammering the backends.
PROVIDER_CONCURRENCY = 1


RESULTS_DIR = PROJECT_ROOT / "results"
STATE_PATH = RESULTS_DIR / "batch_state.json"
LEGACY_STATE_PATH = RESULTS_DIR / "batch_state.json"
OUTPUT_CSV_NAME = "questions_results.csv"
COMBINED_JSON_NAME = "combined.json"
RUN_LOG_NAME = "run_log.txt"

# Return the column order for the CSV file we write out.
def get_output_csv_fieldnames() -> list[str]:
    return [
        "Q#",
        "Graph",
        "Question ID",
        "Question Type",
        "Question",
        "Company",
        "Model",
        "Answer",
        "Correct Answer",
        "Latency",
        "Status",
        "Error",
        "Timestamp",
        "Image Path",
        "CSV Path",
    ]


# Build one row for a single model answer so the results are easy to look at later.
def build_output_row(
    question: dict,
    model: dict,
    result: dict,
    answer_value: str,
    status_value: str,
    error_value: str,
    image_path: Optional[str],
    csv_path: Optional[str],
) -> dict:
    q_id = question["id"]
    metric = question.get("metric") or question.get("question_type") or "Unknown"
    prompt = question.get("prompt", "")
    company = {
        "google": "Google",
        "groq": "Groq",
        "huggingface": "Hugging Face",
        "nvidia": "NVIDIA",
        "openrouter": "OpenRouter",
        "hcompany": "HCompany",
    }.get(model.get("arch"), "Unknown")
    correct_answer = question.get("answer") or question.get("correct_answer") or question.get("expected_answer") or ""

    return {
        "Q#": question.get("question_number", q_id),
        "Graph": question.get("graph", ""),
        "Question ID": q_id,
        "Question Type": question.get("question_type", metric),
        "Question": prompt,
        "Company": company,
        "Model": model["id"],
        "Answer": answer_value,
        "Correct Answer": correct_answer,
        "Latency": result.get("latency_ms", 0),
        "Status": status_value,
        "Error": error_value,
        "Timestamp": datetime.now().isoformat(),
        "Image Path": image_path or "",
        "CSV Path": csv_path or "",
    }


# Clean up some of the noisy provider error text so the logs are easier to read.
def sanitize_error_message(message: str) -> str:
    if not message:
        return ""
    credit_notice = "You have depleted your monthly included credits."
    if credit_notice in message:
        message = message.split(credit_notice)[0].rstrip()
    message = message.replace(
        "Alternatively, subscribe to PRO to get 20x more included usage.",
        "",
    ).strip()
    return message


# Check whether the error looks like a rate-limit or quota issue.
def is_rate_limit_error(error_text: str) -> bool:
    if not error_text:
        return False
    text = error_text.lower()
    return any(token in text for token in [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "quota exceeded",
        "quota exceeded",
        "tokens per day",
        "tpd",
        "too many requests",
        "429",
        "resource exhausted",
        "temporarily unavailable",
    ])


# Make a simple error line that includes the question and the time it happened.
def format_error_entry(question: dict, model_id: str, error_message: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric = question.get("metric", "Unknown")
    question_id = question.get("id", "unknown")
    cleaned = sanitize_error_message(error_message)
    return f"[{timestamp}] [{metric}] [{question_id}] [{model_id}] {cleaned}".strip()


# Clean up the model output so it looks more like a benchmark answer and less like chatty text.
def normalize_answer(response_text: str, prompt: str) -> str:
    if response_text is None:
        return ""

    import re
    from decimal import Decimal, InvalidOperation

    text = str(response_text).strip()
    if not text:
        return ""

    text = text.replace("```", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    def clean_line(line: str) -> str:
        cleaned = re.sub(r"\\boxed\{(.+?)\}", r"\1", line)
        cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
        cleaned = cleaned.replace("`", "").strip()
        cleaned = re.sub(r"^[\-\*]\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.strip(" .,:;-")

    def extract_direct_value(candidate: str) -> str:
        candidate = clean_line(candidate)
        if not candidate:
            return ""

        lowered = candidate.lower()
        if lowered.startswith(("final answer", "the final answer", "answer", "the answer", "result", "the result", "highest", "the highest", "lowest", "the lowest")):
            marker = re.search(r"(?:final\s+answer|answer|result|highest|lowest)\s*(?:is|:)?\s*(.+)$", candidate, flags=re.IGNORECASE)
            if marker:
                candidate = marker.group(1).strip()

        candidate = re.sub(r"^(?:the\s+)?(?:answer|result|highest|lowest)\s*(?:is|:)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip(" .,:;-")
        if not candidate:
            return ""

        if candidate.lower().startswith(("the user", "the request", "here", "i need", "let's", "we", "however", "wait", "but", "actually", "please", "could")):
            return ""

        if len(candidate) > 160:
            return ""

        return candidate

    candidates = []
    for line in lines:
        value = extract_direct_value(line)
        if value:
            candidates.append(value)

    if candidates:
        text = candidates[-1]
    else:
        text = clean_line(lines[-1])

    if not text:
        return ""

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            return format(Decimal(text), ".2f")
        except InvalidOperation:
            return text

    return text


# Reuse the current run folder if there is one already, otherwise make a new one.
def make_run_dir(state: dict) -> Path:
    existing_run_id = state.get("last_run_id")
    if existing_run_id:
        run_dir = RESULTS_DIR / existing_run_id
        if run_dir.exists():
            return run_dir

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = RESULTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# Set up the files right away so an interrupted run can still keep going in the same place.
def initialize_run_artifacts(run_dir: Path, *, batch_size: int, models: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / OUTPUT_CSV_NAME
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=get_output_csv_fieldnames())
            writer.writeheader()

    combined_path = run_dir / COMBINED_JSON_NAME
    if not combined_path.exists():
        payload = {
            "run_id": run_dir.name,
            "total_models": len(models),
            "total_questions": 0,
            "batch_size": batch_size,
            "total_elapsed_s": 0,
            "questions": [],
        }
        with combined_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    log_path = run_dir / RUN_LOG_NAME
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")


# Add this round's results to the shared files without wiping out what already happened.
def append_round_artifacts(
    run_dir: Path,
    *,
    question_result: dict,
    output_rows: list[dict],
    log_lines: list[str],
    models: list[dict],
    batch_size: int,
    run_id: str,
    question: dict,
) -> None:
    initialize_run_artifacts(run_dir, batch_size=batch_size, models=models)

    csv_path = run_dir / OUTPUT_CSV_NAME
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=get_output_csv_fieldnames())
        for row in output_rows:
            writer.writerow(row)

    combined_path = run_dir / COMBINED_JSON_NAME
    if combined_path.exists():
        with combined_path.open("r", encoding="utf-8") as handle:
            existing_combined = json.load(handle)
        questions = existing_combined.get("questions", []) if isinstance(existing_combined.get("questions"), list) else []
        question_exists = any(item.get("question_id") == question_result.get("question_id") for item in questions)
        if not question_exists:
            questions.append(question_result)
        existing_combined["run_id"] = run_id
        existing_combined["total_models"] = len(models)
        existing_combined["total_questions"] = len(questions)
        existing_combined["batch_size"] = batch_size
        existing_combined["questions"] = questions
        with combined_path.open("w", encoding="utf-8") as handle:
            json.dump(existing_combined, handle, indent=2, ensure_ascii=False)

    log_path = run_dir / RUN_LOG_NAME
    existing_log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    existing_entries = {line.strip() for line in existing_log_text.splitlines() if line.strip()}

    for entry in log_lines:
        normalized = entry.strip()
        if normalized and normalized not in existing_entries:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(normalized + "\n")
            existing_entries.add(normalized)

    if question_result.get("responses"):
        ok_count = sum(1 for r in question_result.get("responses", {}).values() if r.get("status") == "ok")
        err_count = len(question_result.get("responses", {})) - ok_count
        if err_count:
            round_line = f"[round] {question.get('id', 'unknown')} ok={ok_count} err={err_count}"
            if round_line not in existing_entries:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(round_line + "\n")


# Give each model a stable provider key so only one call per provider runs at a time.
def get_provider_key(model: dict) -> str:
    if isinstance(model, dict):
        return str(model.get("arch") or "unknown")
    return str(getattr(model, "arch", "unknown") or "unknown")


# Run each provider through a small lock so the requests do not all pile up at once.
async def call_model_safe_limited(
    model: dict,
    prompt: str,
    image_path: str,
    csv_path: Optional[str],
    provider_semaphores: dict[str, asyncio.Semaphore],
) -> dict:
    provider = get_provider_key(model)
    semaphore = provider_semaphores.setdefault(provider, asyncio.Semaphore(PROVIDER_CONCURRENCY))
    async with semaphore:
        return await call_model_safe(model, prompt, image_path, csv_path)

# One model call with timeout and error handling.
# Call one model and always return a result dict, even if it fails.
async def call_model_safe(model: dict, prompt: str, image_path: str, csv_path: Optional[str]) -> dict:
    model_id = model.get("id", "unknown") if isinstance(model, dict) else getattr(model, "id", "unknown")
    start = time.monotonic()
    deadline = start + MODEL_TIMEOUT
    last_error = None

    while True:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            return {
                "status": "error",
                "error": last_error or f"Timeout after {MODEL_TIMEOUT}s",
                "latency_ms": MODEL_TIMEOUT * 1000,
            }

        if remaining_time <= RETRY_DELAY_S:
            return {
                "status": "error",
                "error": last_error or f"Timeout after {MODEL_TIMEOUT}s",
                "latency_ms": round((time.monotonic() - start) * 1000),
            }

        attempt_timeout = max(0.1, remaining_time - RETRY_DELAY_S)
        try:
            call_fn = model.get("call") if isinstance(model, dict) else getattr(model, "call", None)
            if call_fn is None:
                raise AttributeError("Model has no callable 'call' attribute")
            response_text = await asyncio.wait_for(
                call_fn(prompt, image_path, csv_path),
                timeout=attempt_timeout
            )
            elapsed = round((time.monotonic() - start) * 1000)
            return {
                "status": "ok",
                "response": response_text,
                "latency_ms": elapsed,
            }
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "error": f"Timeout after {MODEL_TIMEOUT}s",
                "latency_ms": round((time.monotonic() - start) * 1000),
            }
        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            last_error = error_text
            if is_rate_limit_error(error_text):
                return {
                    "status": "error",
                    "error": error_text,
                    "latency_ms": round((time.monotonic() - start) * 1000),
                }
            await asyncio.sleep(RETRY_DELAY_S)
            continue

    return {
        "status": "error",
        "error": "Unknown error",
        "latency_ms": round((time.monotonic() - start) * 1000),
    }


# One question round where all the enabled models get asked at once.
# Run every model for one question and collect the rows we want to save.
async def run_question_round(
    question: dict,
    models: list,
    run_dir: Path,
    log_lines: list[str],
) -> tuple[dict, list[dict]]:

    q_id = question["id"]
    metric = question.get("metric") or question.get("question_type") or "Unknown"
    prompt = question.get("prompt", "")
    image_path = question.get("image_path") or question.get("image")
    csv_path = question.get("csv_path")

    print(f"\n{'='*70}")
    print(f"  QUESTION: {q_id}  |  Metric: {metric}")
    print(f"  Image:    {image_path}")
    print(f"  Prompt:   {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Sending to {len(models)} models concurrently...")
    print(f"{'='*70}")

    provider_semaphores: dict[str, asyncio.Semaphore] = {}
    tasks = [
        call_model_safe_limited(m, prompt, image_path, csv_path, provider_semaphores)
        for m in models
    ]
    raw_results = await asyncio.gather(*tasks)

    responses = {}
    errors_this_round = []
    output_rows = []

    for model, result in zip(models, raw_results):
        mid = model["id"]
        responses[mid] = result

        if result["status"] == "ok":
            response_text = result["response"] or ""
            answer_value = normalize_answer(response_text, prompt)
            preview = answer_value[:80].replace("\n", " ")
            print(f"  ✅  {mid:<55} {result['latency_ms']:>5}ms  |  {preview}...")
            status_value = "ok"
            error_value = ""
        else:
            print(f"  ❌  {mid:<55}  ERROR: {result['error']}")
            errors_this_round.append(format_error_entry(question, mid, result["error"]))
            answer_value = "Error"
            status_value = "error"
            error_value = result["error"]

        output_rows.append(
            build_output_row(
                question=question,
                model=model,
                result=result,
                answer_value=answer_value,
                status_value=status_value,
                error_value=error_value,
                image_path=image_path,
                csv_path=csv_path,
            )
        )

    if errors_this_round:
        log_lines.extend(errors_this_round)

    question_result = {
        "question_id": q_id,
        "metric": metric,
        "prompt": prompt,
        "image_path": image_path,
        "csv_path": csv_path,
        "graph": question.get("graph"),
        "question_type": question.get("question_type"),
        "timestamp": datetime.now().isoformat(),
        "responses": responses,
    }

    ok_count = sum(1 for r in responses.values() if r["status"] == "ok")
    err_count = len(models) - ok_count
    print(f"\n  Round complete: {ok_count} OK, {err_count} errors")

    return question_result, output_rows


# Main
async def main():
    models = get_active_models()
    questions = get_questions()
    state_path = STATE_PATH if STATE_PATH.exists() or not LEGACY_STATE_PATH.exists() else LEGACY_STATE_PATH
    state = load_state(state_path)

    batch_questions = select_batch_questions(questions, state, batch_size=QUESTIONS_PER_DAY)
    if not batch_questions:
        print("\nNo pending questions remain. All questions have already been completed.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(state)
    run_dir.mkdir(parents=True, exist_ok=True)
    initialize_run_artifacts(run_dir, batch_size=QUESTIONS_PER_DAY, models=models)
    log_lines = []

    print(f"\n{'#'*70}")
    print(f"  SLU CHROME LAB: VARUN'S AI BENCHMARK RUNNER")
    print(f"  Run directory : {run_dir}")
    print(f"  Models        : {len(models)}")
    print(f"  Questions     : {len(questions)}")
    print(f"  Batch size    : {QUESTIONS_PER_DAY}")
    print(f"  Pending batch : {len(batch_questions)}")
    print(f"  Delay between : {INTER_QUESTION_DELAY}s")
    print(f"  Model timeout : {MODEL_TIMEOUT}s")
    print(f"{'#'*70}\n")

    for model in models:
        print(f"  [{model['arch']:<15}] {model['id']}")

    all_results = []
    all_output_rows = []
    total_start = time.time()
    selected_ids = [question["id"] for question in batch_questions]
    print(f"  Selected questions: {', '.join(selected_ids)}")

    for i, question in enumerate(batch_questions):
        result, output_rows = await run_question_round(question, models, run_dir, log_lines)
        all_results.append(result)
        all_output_rows.extend(output_rows)
        append_round_artifacts(
            run_dir,
            question_result=result,
            output_rows=output_rows,
            log_lines=log_lines,
            models=models,
            batch_size=QUESTIONS_PER_DAY,
            run_id=run_dir.name,
            question=question,
        )
        save_state(state_path, state, completed_question_ids=[question["id"]], run_id=run_dir.name)

        if i < len(batch_questions) - 1:
            print(f"\n  ⏳ Waiting {INTER_QUESTION_DELAY}s before next question...")
            await asyncio.sleep(INTER_QUESTION_DELAY)

    total_elapsed = round(time.time() - total_start)

    combined_path = run_dir / COMBINED_JSON_NAME
    if combined_path.exists():
        with combined_path.open("r", encoding="utf-8") as handle:
            existing_combined = json.load(handle)
    else:
        existing_combined = {}

    combined_questions = existing_combined.get("questions", []) if isinstance(existing_combined.get("questions"), list) else []
    for result in all_results:
        if not any(item.get("question_id") == result.get("question_id") for item in combined_questions):
            combined_questions.append(result)

    combined = {
        "run_id": run_dir.name,
        "total_models": len(models),
        "total_questions": len(combined_questions),
        "batch_size": QUESTIONS_PER_DAY,
        "total_elapsed_s": total_elapsed,
        "questions": combined_questions,
    }
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)

    output_csv_path = run_dir / OUTPUT_CSV_NAME
    log_path = run_dir / RUN_LOG_NAME
    total_responses = len(batch_questions) * len(models)
    total_errors = len(log_lines)
    total_ok = total_responses - total_errors

    error_section_lines = []
    if log_lines:
        error_section_lines.append("")
        error_section_lines.append("───---- Errors by Section ─-------")
        grouped_errors = {}
        for entry in log_lines:
            section = "Unknown"
            if " [Section_A] " in entry:
                section = "Section_A"
            elif " [Section_B] " in entry:
                section = "Section_B"
            elif " [Section_C] " in entry:
                section = "Section_C"
            grouped_errors.setdefault(section, []).append(entry)
        for section_name in ["Section_A", "Section_B", "Section_C", "Unknown"]:
            if grouped_errors.get(section_name):
                error_section_lines.append(f"[{section_name}]")
                error_section_lines.extend(grouped_errors[section_name])
                error_section_lines.append("")
    else:
        error_section_lines.append("None")

    summary_lines = [
        f"Run ID        : {run_dir.name}",
        f"Completed at  : {datetime.now().isoformat()}",
        f"Total elapsed : {total_elapsed}s",
        f"Models        : {len(models)}",
        f"Questions     : {len(batch_questions)}",
        f"Responses OK  : {total_ok} / {total_responses}",
        f"Errors        : {total_errors}",
        "",
        "─── Errors ───────────────────────────────────────────",
    ] + error_section_lines

    if log_path.exists():
        existing_log_contents = log_path.read_text(encoding="utf-8")
        if existing_log_contents.strip():
            summary_lines = [existing_log_contents.rstrip(), "", "=" * 60, ""] + summary_lines

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    print(f"\n{'#'*70}")
    print(f"  RUN COMPLETE")
    print(f"  Elapsed       : {total_elapsed}s")
    print(f"  Responses OK  : {total_ok} / {total_responses}")
    print(f"  Errors        : {total_errors}")
    print(f"  Combined JSON : {combined_path}")
    print(f"  Output CSV    : {output_csv_path}")
    print(f"  Log           : {log_path}")
    if log_lines:
        print(f"\n  ⚠️  Models with errors (re-run individually to debug):")
        for line in log_lines:
            print(f"     {line}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    asyncio.run(main())