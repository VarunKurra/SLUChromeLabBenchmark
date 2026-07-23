import json
from datetime import date
from pathlib import Path
from typing import Any, Optional, Union


# Load the saved scheduler state from disk. If nothing is there yet, use a simple default.
def load_state(state_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    path = Path(state_path) if state_path is not None else Path("results") / "batch_state.json"
    if not path.exists():
        return {
            "completed_question_ids": [],
            "last_run_date": None,
            "completed_batches": 0,
        }

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError:
        return {
            "completed_question_ids": [],
            "last_run_date": None,
            "completed_batches": 0,
        }

    return {
        "completed_question_ids": data.get("completed_question_ids", []),
        "last_run_date": data.get("last_run_date"),
        "completed_batches": data.get("completed_batches", 0),
    }


# Save the current progress so the run can pick up where it left off later.
def save_state(
    state_path: Union[str, Path],
    state: dict[str, Any],
    *,
    completed_question_ids: Optional[list[str]] = None,
    run_id: Optional[str] = None,
) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    completed = state.get("completed_question_ids", [])
    if completed_question_ids is not None:
        for question_id in completed_question_ids:
            if question_id not in completed:
                completed.append(question_id)

    state["completed_question_ids"] = completed
    state["last_run_date"] = state.get("last_run_date") or date.today().isoformat()
    state["completed_batches"] = state.get("completed_batches", 0)

    payload = {
        "completed_question_ids": completed,
        "last_run_date": state["last_run_date"],
        "completed_batches": state["completed_batches"],
    }
    if run_id is not None:
        payload["last_run_id"] = run_id
        state["last_run_id"] = run_id

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


# Pick the next set of questions that still need to be done.
def select_batch_questions(
    questions: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    completed_ids = set(state.get("completed_question_ids", []))
    pending_questions = [q for q in questions if q.get("id") not in completed_ids]
    return pending_questions[: max(0, batch_size)]
