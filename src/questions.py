# This file pulls questions from the CSV and also finds the matching graph files.
# The runner expects a CSV in the project data folder, and each row has the question,
# its graph name, and a bit of extra info about what kind of chart it is.

import csv
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# This is turned on so the CSV is the main source for questions.
USE_CSV = True
CSV_PATH = str(DATA_DIR / "questions.csv")

# Fallback image for test runs.
DEFAULT_IMAGE = "/Users/varun/Downloads/Test_Graph.webp"

# A tiny backup list in case the CSV is missing.
QUESTIONS = [
    {
        "id": "sec_a_q1",
        "metric": "Section_A",
        "prompt": "Describe the graph.",
        "image": DEFAULT_IMAGE,
    }
]


# Make graph names easier to compare by turning them into a simple lowercase form.
def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


# Try a few likely folders until the matching graph asset is found.
def _resolve_graph_asset(graph_name: str, base_dir: Path, extension: str) -> Optional[str]:
    if not graph_name:
        return None

    target = _normalize_name(graph_name)
    candidate_roots = []
    for root in [base_dir, PROJECT_ROOT]:
        candidate_roots.extend([
            root,
            root / "Graphs",
            root / "assets" / "graphs",
        ])

    seen_roots = set()
    for root in candidate_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)

        for graph_type in ["Bar", "Line", "Pie", "Scatter"]:
            folder_candidates = [
                root / graph_type,
                root / "Graphs" / graph_type,
                root / "assets" / "graphs" / graph_type,
            ]
            for folder in folder_candidates:
                if not folder.exists():
                    continue

                for path in folder.glob(f"*{extension}"):
                    if not path.is_file():
                        continue
                    normalized_stem = _normalize_name(path.stem)
                    if normalized_stem == target or target in normalized_stem:
                        return str(path.resolve())

    return None


# Load questions from the CSV format used for the benchmark sheet.
def load_questions_from_csv(path: str, base_dir: Optional[Path] = None) -> list[dict]:
    questions = []
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    base = Path(base_dir) if base_dir is not None else csv_path.parent
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            q_number = (row.get("Q#") or row.get("Q") or "").strip()
            graph = (row.get("Graph") or "").strip()
            question_type = (row.get("Question Type") or row.get("QuestionType") or "").strip()
            prompt = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()

            question_id = f"q{q_number}" if q_number else f"q{index}"
            image_path = _resolve_graph_asset(graph, base, ".png")
            csv_path_for_question = _resolve_graph_asset(graph, base, ".csv")

            questions.append({
                "id": question_id,
                "question_number": q_number,
                "graph": graph,
                "question_type": question_type,
                "metric": question_type or "Unknown",
                "prompt": prompt,
                "answer": answer,
                "image": image_path,
                "image_path": image_path,
                "csv_path": csv_path_for_question,
                "source_csv": str(csv_path),
            })
    return questions


# Return the current question list from the CSV when it exists, or fall back to the built-in backup list.
def get_questions() -> list[dict]:
    root_dir = PROJECT_ROOT
    csv_path = Path(CSV_PATH)
    if not csv_path.is_absolute():
        csv_path = root_dir / csv_path

    if USE_CSV and csv_path.exists():
        print(f"[questions] Loading from CSV: {csv_path}")
        return load_questions_from_csv(str(csv_path), base_dir=root_dir)

    if USE_CSV and not csv_path.exists():
        print(f"[questions] CSV not found at {csv_path}; falling back to built-in questions")

    return QUESTIONS
