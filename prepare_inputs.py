"""Convert downloaded Spider or BIRD questions to EDM-SQL inference JSONL."""

import argparse
import json
import os
from pathlib import Path


LAYOUTS = {
    "spider-dev": ("dev.json", "database"),
    "spider-test": ("test.json", "test_database"),
    "bird-dev": ("dev.json", "dev_databases"),
}


def prepare_inputs(dataset: str, data_dir: Path, output: Path) -> int:
    questions_file, database_dir = LAYOUTS[dataset]
    data_dir = data_dir.resolve()
    output = output.absolute()
    source = data_dir / questions_file
    rows = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a nonempty JSON array in {source}")

    records, seen_ids, database_paths = [], set(), {}
    for index, row in enumerate(rows):
        label = f"{source} record {index + 1}"
        if not isinstance(row, dict):
            raise ValueError(f"{label}: expected a JSON object")
        question, db_id = row.get("question"), row.get("db_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{label}: missing question")
        if (not isinstance(db_id, str) or not db_id.strip() or db_id in (".", "..")
                or any(char in db_id for char in "/\\:")):
            raise ValueError(f"{label}: invalid db_id")
        record_id = row.get("question_id", index)
        if type(record_id) not in (int, str) or not str(record_id).strip():
            raise ValueError(f"{label}: invalid question_id")
        if str(record_id) in seen_ids:
            raise ValueError(f"{label}: duplicate question_id {record_id}")
        seen_ids.add(str(record_id))

        if db_id not in database_paths:
            database = data_dir / database_dir / db_id / f"{db_id}.sqlite"
            if not database.is_file():
                raise FileNotFoundError(f"{label}: SQLite database not found: {database}")
            try:
                database_paths[db_id] = os.path.relpath(database, output.parent)
            except ValueError:
                database_paths[db_id] = str(database)

        record = {"id": record_id, "question": question, "database": database_paths[db_id]}
        if dataset == "bird-dev" and row.get("evidence") is not None:
            if not isinstance(row["evidence"], str):
                raise ValueError(f"{label}: evidence must be text")
            record["evidence"] = row["evidence"]
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            print(json.dumps(record, ensure_ascii=False), file=stream)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=LAYOUTS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Root of the extracted dataset")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; existing files are not overwritten")
    args = parser.parse_args()
    try:
        count = prepare_inputs(args.dataset, args.data_dir, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote {count} questions to {args.output}")


if __name__ == "__main__":
    main()
