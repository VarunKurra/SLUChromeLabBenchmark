#!/usr/bin/env python3
import csv
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional, Tuple


# Find the newest questions_results.csv file under the results folder.
def find_latest_questions_results(results_dir: Path) -> Optional[Path]:
    candidates = sorted(results_dir.glob("*/questions_results.csv"))
    if not candidates:
        return None
    return candidates[-1]


# Check whether the value looks like a plain number.
def is_numeric_like(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    text = text.replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text))


# Turn a string into a Decimal when it is safe to do so.
def parse_decimal(value: str) -> Optional[Decimal]:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


# Make text comparisons a bit more forgiving by stripping extra spacing and punctuation.
def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", "", text)
    # Make a few common dash and punctuation cases look the same.
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+([\-.,:;!?()\[\]{}\/\\])", r"\1", text)
    text = re.sub(r"([\-.,:;!?()\[\]{}\/\\])\s+", r"\1", text)
    return text


# Grade one row by comparing the answer to the expected one.
def grade_row(row: dict) -> int:
    question_type = (row.get("Question Type") or "").strip()
    status = (row.get("Status") or "").strip()

    if question_type not in {"Comprehension", "Interpretation"}:
        return ""

    if status != "ok":
        return 0

    answer = row.get("Answer") or ""
    correct_answer = row.get("Correct Answer") or ""

    if not str(answer).strip() and not str(correct_answer).strip():
        return 1

    if is_numeric_like(answer) and is_numeric_like(correct_answer):
        answer_value = parse_decimal(answer)
        correct_value = parse_decimal(correct_answer)
        if answer_value is not None and correct_value is not None:
            return 1 if answer_value == correct_value else 0

    return 1 if normalize_text(answer) == normalize_text(correct_answer) else 0


# Add a correctness column to the rows before writing the CSV out.
def build_output_rows(rows: List[dict]) -> List[dict]:
    output_rows: List[dict] = []
    for row in rows:
        new_row = dict(row)
        new_row["Correct?"] = grade_row(row)
        output_rows.append(new_row)
    return output_rows


# Write the graded rows out to a new CSV file.
def write_csv(input_path: Path, output_path: Path, rows: List[dict]) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        if "Correct?" not in fieldnames:
            insert_at = fieldnames.index("Correct Answer") + 1 if "Correct Answer" in fieldnames else len(fieldnames)
            fieldnames.insert(insert_at, "Correct?")

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                new_row = {}
                for field in fieldnames:
                    if field == "Correct?":
                        new_row[field] = row.get("Correct?", "")
                    else:
                        new_row[field] = row.get(field, "")
                writer.writerow(new_row)


# Run the grading flow from the command line.
def main() -> int:
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent
        results_dir = repo_root / "results"
        input_path = find_latest_questions_results(results_dir)
        if input_path is None:
            print("No questions_results.csv found under results/", file=sys.stderr)
            return 1

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = input_path.with_name("questions_results_graded.csv")

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    graded_rows = build_output_rows(rows)
    write_csv(input_path, output_path, graded_rows)

    print(f"Wrote {len(graded_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
