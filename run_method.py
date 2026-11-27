"""Run the EDM-SQL reference code on JSON Lines input."""

import argparse
import json
import os
import sys
from pathlib import Path

from edm_sql.value_linking import LocalEmbedding, ValueIndex, add_index_arguments


def add_feedback_arguments(parser):
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--memory", action="store_true",
                       help="Enable memory with interactive human feedback")
    modes.add_argument("--no-memory", action="store_false", dest="memory",
                       help="Disable memory (default)")
    parser.set_defaults(memory=False)


def interactive_feedback(record_id, database, question):
    started = False
    labels = {"1": True, "0": False, "s": None}

    def verdict(stage, selected_sql):
        nonlocal started
        if not started:
            print(f"\nFeedback for {record_id} ({database.name})\nQuestion: {question}", file=sys.stderr)
            started = True
        print(f"{stage} Level-3 selected SQL:\n{selected_sql}", file=sys.stderr)
        while True:
            print("[1] Correct  [0] Incorrect  [s] Skip: ", end="", file=sys.stderr, flush=True)
            try:
                choice = input().strip().lower()
            except EOFError as exc:
                raise ValueError("Human feedback input closed. Use 's' to skip or run without --memory.") from exc
            if choice in labels:
                return labels[choice]
            print("Enter 1, 0, or s.", file=sys.stderr)

    return verdict


def read_question_rows(path):
    root = path.resolve().parent
    with path.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {index + 1}: invalid JSON ({exc.msg})") from exc
                if (not isinstance(row, dict) or not isinstance(row.get("database"), str)
                        or not row["database"].strip()):
                    raise ValueError(f"Line {index + 1}: expected a JSON object with a database path")
                yield str(row.get("id", index)), row, root / row["database"]


def completion_from_env():
    missing = [key for key in ("EDM_SQL_API_KEY", "EDM_SQL_BASE_URL", "EDM_SQL_MODEL") if not os.environ.get(key)]
    if missing:
        raise ValueError("Missing provider configuration: " + ", ".join(missing))

    from openai import OpenAI
    from edm_sql.prompts import messages_for

    client = OpenAI(api_key=os.environ["EDM_SQL_API_KEY"], base_url=os.environ["EDM_SQL_BASE_URL"])
    model = os.environ["EDM_SQL_MODEL"]

    def complete(role, payload):
        response = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=messages_for(role, payload),
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```") and content.endswith("```"):
            content = content.partition("\n")[2].removesuffix("```").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON Lines input, in the order to answer")
    add_feedback_arguments(parser)
    add_index_arguments(parser)
    args = parser.parse_args()

    from edm_sql.database import Database
    from edm_sql.pipeline import EDMSQL

    try:
        complete = completion_from_env()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    embed = LocalEmbedding(args.embedding_model, args.embedding_device)
    method = EDMSQL(complete, enable_memory=args.memory, embed=embed,
                    value_index=ValueIndex(args.value_index_dir, embed.model_name))
    for question_id, row, database in read_question_rows(args.input):
        try:
            feedback = interactive_feedback(question_id, database, row["question"]) if args.memory else None
            result = method.answer(
                row["question"], Database(database), schema=row.get("schema"),
                initial_sql=row.get("initial_sql"), schema_sql=row.get("schema_sql"),
                generation_inputs=row,
                schema_context=row.get("augmented_schema"),
                feedback=feedback,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"Question {question_id}: {exc}\n")
        print(json.dumps({"id": question_id, **result}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
