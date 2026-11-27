import io
import json
import sqlite3
import sys
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from edm_sql.pipeline import EDMSQL
from run_method import main as method_main
from run_selector import main as selector_main, select_record


CURRENT = "SELECT name FROM items WHERE id = 1"
NEW = "SELECT name FROM items WHERE id = 2"


class SelectionToolsTests(unittest.TestCase):
    def fixture(self, suffix):
        path = Path(__file__).parent / f"selection-test-{uuid4().hex}{suffix}"
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def database(self):
        path = self.fixture(".sqlite")
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.executescript("CREATE TABLE items(id INTEGER, name TEXT);"
                                     "INSERT INTO items VALUES (1, 'Alpha'), (2, 'Beta');")
        return path

    def test_cli_runs_level_one_without_a_model_or_index(self):
        path = self.fixture(".jsonl")
        path.write_text(json.dumps({"id": "demo", "stage": "RSM", "question": "Find Beta.",
                                    "database": self.database().name, "current_sql": CURRENT, "new_sql": CURRENT}),
                        encoding="utf-8")
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_selector.py", str(path), "--no-memory"]), \
                patch("run_selector.completion_from_env", side_effect=AssertionError("unexpected model call")), \
                redirect_stdout(output):
            selector_main()
        record = json.loads(output.getvalue())
        self.assertEqual((record["id"], record["stage"], record["level"], record["selected_sql"]),
                         ("demo", "RSM", 1, CURRENT))

    def test_selector_supplies_evidence_and_updates_only_after_the_decision(self):
        roles = []
        path = self.database()

        def complete(role, payload):
            roles.append(role)
            if role == "adjudicate":
                self.assertEqual(payload["question_evidence"], "Beta is item 2.")
                return {"verdict": "accept", "defect": "Wrong ID", "evidence": "Question requests Beta"}
            if role == "summarize":
                self.assertTrue(payload["outcome_verdict"])
                self.assertEqual(payload["question_evidence"], "Beta is item 2.")
                self.assertIsInstance(payload["evidence"], dict)
                return {"card": {"applies_when": "ID evidence", "risk": "wrong row", "guidance": "use the stated ID"}}
            self.assertEqual(role, "update")
            return {"pool": [payload["card"]]}

        def feedback(stage, sql):
            roles.append("feedback")
            self.assertEqual((stage, sql), ("SSM", NEW))
            return True

        method = EDMSQL(complete)
        decision = select_record(method, {"stage": "SSM", "question": "Find Beta.",
                                         "evidence": "Beta is item 2.",
                                         "current_sql": CURRENT, "new_sql": NEW}, path, feedback)
        self.assertEqual((decision["level"], decision["selected_sql"]), (3, NEW))
        self.assertEqual(roles, ["adjudicate", "feedback", "summarize", "update"])
        self.assertEqual((len(method.memory["SSM"]), len(method.memory["RSM"])), (1, 0))

    def test_both_clis_reject_non_interactive_feedback_options(self):
        options = (("--feedback", "verdicts.json"), ("--feedback-provider", "demo_oracle:evaluate"))
        for command, main in (("run_method.py", method_main), ("run_selector.py", selector_main)):
            for flag, value in options:
                with self.subTest(command=command, flag=flag):
                    errors = io.StringIO()
                    with patch.object(sys, "argv", [command, "input.jsonl", flag, value]), \
                            redirect_stderr(errors), self.assertRaises(SystemExit) as exit_info:
                        main()
                    self.assertEqual(exit_info.exception.code, 2)
                    self.assertIn(f"unrecognized arguments: {flag}", errors.getvalue())

    def test_human_feedback_waits_after_level_three_and_guides_next_pair(self):
        path = self.fixture(".jsonl")
        database = self.database()
        records = [dict(id=key, stage="SSM", question="Find Beta.", database=database.name,
                        current_sql=CURRENT, new_sql=NEW if key != "l1" else CURRENT)
                   for key in ("first", "l1", "second")]
        path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
        for flags, expected in (([], ["check:0", "check:0"]),
                                (["--memory"], ["check:0", "human", "human", "summarize", "update",
                                                "check:1", "human"])):
            with self.subTest(flags=flags):
                events, answers = [], iter(("invalid", "1", "s"))

                def read_answer():
                    events.append("human")
                    return next(answers)

                def complete(role, payload):
                    if role == "adjudicate":
                        events.append(f"check:{len(payload['memory_pool'])}")
                        return {"verdict": "accept", "defect": "Wrong ID", "evidence": "Beta is item 2"}
                    events.append(role)
                    if role == "summarize":
                        self.assertIs(payload["outcome_verdict"], True)
                        return {"card": {"applies_when": "ID evidence", "risk": "wrong row", "guidance": "check ID"}}
                    return {"pool": [payload["card"]]}

                output, prompts = io.StringIO(), io.StringIO()
                with patch.object(sys, "argv", ["run_selector.py", str(path), *flags]), \
                        patch("run_selector.completion_from_env", return_value=complete), \
                        patch("builtins.input", side_effect=read_answer), \
                        redirect_stdout(output), redirect_stderr(prompts):
                    selector_main()
                self.assertEqual(events, expected)
                self.assertEqual([json.loads(line)["selected_sql"] for line in output.getvalue().splitlines()],
                                 [NEW, CURRENT, NEW])
                if flags:
                    self.assertIn(NEW, prompts.getvalue())
                    self.assertNotIn("Feedback for l1", prompts.getvalue())
                else:
                    self.assertEqual(prompts.getvalue(), "")
