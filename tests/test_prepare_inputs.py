import json
import unittest
from pathlib import Path
from uuid import uuid4

from prepare_inputs import prepare_inputs
from run_method import read_question_rows


class PrepareInputsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / f"input-test-{uuid4().hex}"
        self.make_dir(self.root)
        self.data = self.make_dir(self.root / "data")
        self.output_dir = self.make_dir(self.root / "runs")

    def make_dir(self, path):
        path.mkdir()
        self.addCleanup(path.rmdir)
        return path

    def write_json(self, filename, rows):
        source = self.data / filename
        source.write_text(json.dumps(rows), encoding="utf-8")
        self.addCleanup(source.unlink)

    def add_database(self, layout):
        parent = self.make_dir(self.make_dir(self.data / layout) / "demo")
        database = parent / "demo.sqlite"
        database.touch()
        self.addCleanup(database.unlink)
        return database

    def output_path(self, name):
        path = self.output_dir / name
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_spider_dev_and_test_use_their_own_sqlite_directories(self):
        for split, directory in (("dev", "database"), ("test", "test_database")):
            with self.subTest(split=split):
                database = self.add_database(directory)
                self.write_json(f"{split}.json", [
                    {"question": "First", "db_id": "demo", "query": "gold SQL", "sql": {"select": []}},
                    {"question": "Second", "db_id": "demo", "query": "another gold SQL"},
                ])
                output = self.output_path(f"{split}.jsonl")
                self.assertEqual(prepare_inputs(f"spider-{split}", self.data, output), 2)
                records = list(read_question_rows(output))
                self.assertEqual([(key, row["question"]) for key, row, _ in records],
                                 [("0", "First"), ("1", "Second")])
                for _, row, resolved in records:
                    self.assertEqual(set(row), {"id", "question", "database"})
                    self.assertFalse(Path(row["database"]).is_absolute())
                    self.assertEqual(resolved.resolve(), database.resolve())

    def test_bird_preserves_ids_order_and_separate_evidence_without_gold(self):
        database = self.add_database("dev_databases")
        self.write_json("dev.json", [
            {"question_id": 12, "question": "First", "db_id": "demo", "evidence": "Use the given units.",
             "SQL": "gold SQL", "gold_sql": "other reference"},
            {"question_id": 5, "question": "Second", "db_id": "demo", "evidence": "", "SQL": "gold SQL"},
        ])
        output = self.output_path("bird.jsonl")
        self.assertEqual(prepare_inputs("bird-dev", self.data, output), 2)
        records = list(read_question_rows(output))
        self.assertEqual([(key, row["question"], row["evidence"]) for key, row, _ in records],
                         [("12", "First", "Use the given units."), ("5", "Second", "")])
        for _, row, resolved in records:
            self.assertEqual(set(row), {"id", "question", "database", "evidence"})
            self.assertEqual(resolved.resolve(), database.resolve())

    def test_missing_database_leaves_no_output(self):
        self.write_json("dev.json", [{"question": "First", "db_id": "missing"}])
        output = self.output_path("missing.jsonl")
        with self.assertRaisesRegex(FileNotFoundError, "record 1: SQLite database not found"):
            prepare_inputs("spider-dev", self.data, output)
        self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        self.add_database("database")
        self.write_json("dev.json", [{"question": "First", "db_id": "demo"}])
        output = self.output_path("existing.jsonl")
        output.write_text("keep this file", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            prepare_inputs("spider-dev", self.data, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep this file")
