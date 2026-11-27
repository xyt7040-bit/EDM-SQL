"""Run three-level EDM-SQL selection on supplied candidate pairs, without generation."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from edm_sql.database import Database
from edm_sql.pipeline import EDMSQL, STAGES
from run_method import add_feedback_arguments, completion_from_env, interactive_feedback, read_question_rows


def select_record(method, row, database, feedback=None):
    for field in ("stage", "question", "current_sql", "new_sql"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")
    stage = row["stage"].strip().upper()
    if stage not in STAGES:
        raise ValueError("stage must be SSM or RSM")
    db = Database(database)
    context = {"question": row["question"], "question_evidence": row.get("evidence"),
               "schema": row.get("schema") or db.schema(), "augmented_schema": row.get("augmented_schema")}
    decision = method.select_pair(stage, context, db, row["current_sql"], row["new_sql"])
    method.close_question(row["question"], (decision,), feedback,
                          question_evidence=context["question_evidence"])
    return asdict(decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Candidate pairs in JSON Lines format, evaluated in file order")
    add_feedback_arguments(parser)
    args = parser.parse_args()

    backend = None

    def complete(role, payload):
        nonlocal backend
        if backend is None:
            backend = completion_from_env()
        return backend(role, payload)

    try:
        method = EDMSQL(complete, enable_memory=args.memory)
        for pair_id, row, database in read_question_rows(args.input):
            try:
                feedback = interactive_feedback(pair_id, database, row.get("question")) if args.memory else None
                result = select_record(method, row, database, feedback)
            except (OSError, ValueError) as exc:
                parser.exit(2, f"Pair {pair_id}: {exc}\n")
            print(json.dumps({"id": pair_id, **result}, ensure_ascii=False, default=str), flush=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
