"""Rescore the saved EDM-SQL predictions with the bundled evaluators."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "spider-dev": ("predict.txt", 1034),
    "bird-dev": ("predict_dev.json", 1534),
}
BIRD_SEPARATOR = "\t----- bird -----\t"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"Missing input file: {path}")
    return path


def require_directory(path: Path) -> Path:
    if not path.is_dir():
        raise ValueError(f"Missing database directory: {path}")
    return path


def read_rows(path: Path, expected: int) -> list[str]:
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} lines in {path}, found {len(rows)}")
    for line, row in enumerate(rows, start=1):
        if not row.strip():
            raise ValueError(f"Blank record in {path} at line {line}; expected one record per question")
    return rows


def run_evaluator(dataset: str, *arguments: str | Path) -> None:
    script = ROOT / "evaluators" / dataset / "evaluation.py"
    subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(script), *map(str, arguments)],
        check=True,
    )


def spider(data: Path, predictions: Path, expected: int) -> None:
    gold = require_file(data / "dev_gold.sql")
    tables = require_file(data / "tables.json")
    database = require_directory(data / "database")
    read_rows(gold, expected)
    # The upstream Spider evaluator treats blank lines as session boundaries.
    read_rows(predictions, expected)
    run_evaluator(
        "spider",
        "--gold", gold, "--pred", predictions, "--db", database,
        "--table", tables, "--etype", "exec",
    )


def bird_database_ids(predictions: object, gold_rows: list[str]) -> set[str]:
    expected = len(gold_rows)
    if not isinstance(predictions, dict) or list(predictions) != [str(i) for i in range(expected)]:
        raise ValueError(f"Expected ordered BIRD prediction IDs 0 through {expected - 1}")
    database_ids = set()
    for index, gold in enumerate(gold_rows):
        _, delimiter, gold_db = gold.strip().rpartition("\t")
        if not delimiter or not gold_db:
            raise ValueError(f"Missing reference database ID for BIRD prediction {index}")
        row = predictions[str(index)]
        if not isinstance(row, str) or row.count(BIRD_SEPARATOR) != 1:
            raise ValueError(f"Invalid BIRD prediction {index}; expected SQL and a database ID")
        _, predicted_db = row.split(BIRD_SEPARATOR)
        if predicted_db != gold_db:
            raise ValueError(
                f"BIRD database mismatch at prediction {index}: "
                f"prediction uses {predicted_db!r}, reference uses {gold_db!r}"
            )
        database_ids.add(gold_db)
    return database_ids


def bird(data: Path, predictions: Path, expected: int, workers: int) -> None:
    gold = data / "dev_gold.sql"
    if not gold.is_file():
        gold = require_file(data / "dev.sql")
    gold_rows = read_rows(gold, expected)
    database = require_directory(data / "dev_databases")
    difficulty = require_file(data / "dev.json")
    rows = json.loads(predictions.read_text(encoding="utf-8-sig"))
    for db_id in sorted(bird_database_ids(rows, gold_rows)):
        require_file(database / db_id / f"{db_id}.sqlite")
    run_evaluator(
        "bird",
        "--predicted_sql_path", predictions, "--ground_truth_path", gold,
        "--db_root_path", str(database) + os.sep,
        "--diff_json_path", difficulty, "--data_mode", "dev",
        "--num_cpus", str(workers), "--meta_time_out", "30",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root of the original dev dataset")
    parser.add_argument("--pred", type=Path, help="Override the bundled prediction file")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for BIRD")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    filename, expected = DATASETS[args.dataset]
    try:
        predictions = require_file(args.pred or ROOT / "results" / args.dataset / filename).resolve()
        data = args.data_dir.resolve()
        if args.dataset == "spider-dev":
            spider(data, predictions, expected)
        else:
            bird(data, predictions, expected, args.workers)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
