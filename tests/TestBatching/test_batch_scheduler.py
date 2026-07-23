import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from src.batch_scheduler import load_state, save_state, select_batch_questions
from src.questions import load_questions_from_csv
from src.models import _build_prompt
from src.runner import (
    call_model_safe,
    format_error_entry,
    get_output_csv_fieldnames,
    normalize_answer,
    sanitize_error_message,
)


class BatchSchedulerTests(unittest.TestCase):
    def test_sanitize_error_message_removes_credit_notice(self):
        raw_message = (
            "HfHubHTTPError: 402 Client Error\n"
            "You have depleted your monthly included credits. Purchase pre-paid credits to continue using Inference Providers. "
            "Alternatively, subscribe to PRO to get 20x more included usage."
        )
        cleaned = sanitize_error_message(raw_message)
        self.assertNotIn("You have depleted your monthly included credits", cleaned)
        self.assertNotIn("Alternatively, subscribe to PRO", cleaned)

    def test_format_error_entry_includes_timestamp_section_and_question(self):
        entry = format_error_entry(
            {"id": "sec_a_q1", "metric": "Section_A"},
            "gemini-test",
            "boom",
        )
        self.assertIn("Section_A", entry)
        self.assertIn("sec_a_q1", entry)
        self.assertIn("gemini-test", entry)
        self.assertRegex(entry, r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")

    def test_select_batch_questions_skips_completed_and_respects_batch_size(self):
        questions = [
            {"id": "q1"},
            {"id": "q2"},
            {"id": "q3"},
            {"id": "q4"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = load_state(state_path)
            state["completed_question_ids"] = ["q1", "q3"]
            save_state(state_path, state)

            selected = select_batch_questions(questions, state, batch_size=2)

            self.assertEqual([q["id"] for q in selected], ["q2", "q4"])

    def test_save_state_persists_completed_question_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = load_state(state_path)
            save_state(state_path, state, completed_question_ids=["q1", "q2"], run_id="run-1")
            save_state(state_path, state, completed_question_ids=["q3"], run_id="run-2")

            with state_path.open("r", encoding="utf-8") as fh:
                persisted = json.load(fh)

            self.assertEqual(persisted["completed_question_ids"], ["q1", "q2", "q3"])

    def test_get_output_csv_fieldnames_puts_question_id_before_question_type(self):
        fieldnames = get_output_csv_fieldnames()

        self.assertNotIn("Prompt", fieldnames)
        self.assertLess(fieldnames.index("Question ID"), fieldnames.index("Question Type"))

    def test_load_questions_from_csv_parses_graph_and_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "Questions.csv"
            csv_path.write_text(
                "Q#,Graph,Question Type,Question\n"
                "2,Line2,Comprehension,What trend is shown?\n",
                encoding="utf-8",
            )
            graphs_dir = Path(tmpdir) / "Graphs" / "Line"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            (graphs_dir / "line2.png").write_bytes(b"png")
            (graphs_dir / "line2.csv").write_text("x,y\n1,2\n", encoding="utf-8")

            questions = load_questions_from_csv(str(csv_path), base_dir=Path(tmpdir))

            self.assertEqual(questions[0]["id"], "q2")
            self.assertEqual(questions[0]["graph"], "Line2")
            self.assertEqual(questions[0]["question_type"], "Comprehension")
            self.assertEqual(questions[0]["prompt"], "What trend is shown?")
            self.assertTrue(questions[0]["image_path"].endswith("Graphs/Line/line2.png"))
            self.assertTrue(questions[0]["csv_path"].endswith("Graphs/Line/line2.csv"))

    def test_normalize_answer_extracts_numeric_value_from_verbose_response(self):
        cleaned = normalize_answer(
            "The concentration is 112.0.\nThe request asks for exactly two decimal places.\n112.00",
            "At hour 8, what is the concentration of N.O2?",
        )
        self.assertEqual(cleaned, "112.00")

    def test_normalize_answer_uses_last_final_answer_marker(self):
        cleaned = normalize_answer(
            "Some earlier reasoning\nFINAL ANSWER: 19.00\nMore thought\nFINAL ANSWER: 18.93",
            "At what hour does the concentration of NMHC first reach 450?",
        )
        self.assertEqual(cleaned, "18.93")

    def test_normalize_answer_formats_integer_outputs_to_two_decimals(self):
        cleaned = normalize_answer("FINAL ANSWER: 18", "At what hour does the concentration of NMHC first reach 450?")
        self.assertEqual(cleaned, "18.00")

    def test_normalize_answer_prefers_final_numeric_answer_from_verbose_response(self):
        cleaned = normalize_answer(
            "The user wants to find the hour when the concentration of NMHC first reaches 450.\n"
            "The concentration reaches 450 between hour 18 and 19.\n"
            "The final answer is: 19.00\n"
            "However, in",
            "At what hour does the concentration of NMHC first reach 450?",
        )
        self.assertEqual(cleaned, "19.00")

    def test_build_prompt_requires_single_line_final_answer_format(self):
        prompt = _build_prompt("At what hour does the concentration of NMHC first reach 450?", "/tmp/graph.csv")

        self.assertIn("Return ONLY the final answer", prompt)
        self.assertIn("Do not include reasoning", prompt)
        self.assertIn("exactly two decimal places", prompt)

    def test_call_model_safe_retries_gemma_on_503_unavailable(self):
        class FailingThenWorkingModel:
            def __init__(self):
                self.calls = 0

            async def __call__(self, prompt, image_path, csv_path=None):
                self.calls += 1
                if self.calls == 1:
                    raise Exception("ServerError: 503 UNAVAILABLE")
                return "19.00"

        model = {"id": "google/gemma-4-31b-it", "call": FailingThenWorkingModel()}
        result = asyncio.run(call_model_safe(model, "test", "img.png", None))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["response"], "19.00")


if __name__ == "__main__":
    unittest.main()
