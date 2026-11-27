import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import BIRD_SEPARATOR, bird_database_ids, read_rows


class EvaluationInputTests(unittest.TestCase):
    def test_spider_rejects_blank_prediction_at_its_original_position(self):
        with patch.object(Path, "read_text", return_value="SELECT 1\n \nSELECT 2\n"):
            with self.assertRaisesRegex(ValueError, r"Blank record.*at line 2"):
                read_rows(Path("predict.txt"), 3)

    def test_spider_accepts_trailing_newline(self):
        with patch.object(Path, "read_text", return_value="SELECT 1\n"):
            self.assertEqual(read_rows(Path("predict.txt"), 1), ["SELECT 1"])

    def test_spider_rejects_wrong_count(self):
        with patch.object(Path, "read_text", return_value="SELECT 1\n"):
            with self.assertRaisesRegex(ValueError, "Expected 2 lines"):
                read_rows(Path("predict.txt"), 2)

    def test_bird_accepts_aligned_records(self):
        predictions = {"0": f"SELECT 1{BIRD_SEPARATOR}first", "1": f"SELECT 2{BIRD_SEPARATOR}second"}
        self.assertEqual(
            bird_database_ids(predictions, ["SELECT 1\tfirst", "SELECT 2\tsecond"]),
            {"first", "second"},
        )

    def test_bird_rejects_mismatched_database(self):
        predictions = {"0": f"SELECT 1{BIRD_SEPARATOR}wrong_db"}
        with self.assertRaisesRegex(ValueError, "database mismatch at prediction 0"):
            bird_database_ids(predictions, ["SELECT 1\tright_db"])

    def test_bird_rejects_ids_out_of_order(self):
        predictions = {"1": f"SELECT 1{BIRD_SEPARATOR}db", "0": f"SELECT 2{BIRD_SEPARATOR}db"}
        with self.assertRaisesRegex(ValueError, "ordered BIRD prediction IDs"):
            bird_database_ids(predictions, ["SELECT 1\tdb", "SELECT 2\tdb"])

    def test_bird_rejects_missing_database_separator(self):
        with self.assertRaisesRegex(ValueError, "Invalid BIRD prediction 0"):
            bird_database_ids({"0": "SELECT 1"}, ["SELECT 1\tdb"])


if __name__ == "__main__":
    unittest.main()
