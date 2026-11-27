"""Build local database-value indexes before running EDM-SQL."""

import argparse
from pathlib import Path

from edm_sql.database import Database
from edm_sql.value_linking import LocalEmbedding, ValueIndex, add_index_arguments
from run_method import read_question_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="The same JSON Lines input used by run_method.py")
    parser.add_argument("--force", action="store_true", help="Rebuild indexes that already match")
    add_index_arguments(parser)
    args = parser.parse_args()

    embed = LocalEmbedding(args.embedding_model, args.embedding_device)
    index = ValueIndex(args.value_index_dir, embed.model_name)
    seen = set()
    for _, row, database in read_question_rows(args.input):
        if row.get("augmented_schema") is not None or row.get("schema_sql") is not None:
            continue
        db = Database(database, timeout=600)
        if db.path in seen:
            continue
        seen.add(db.path)
        if not args.force:
            try:
                path = index.require(db)
            except (FileNotFoundError, ValueError):
                pass
            else:
                print(f"Reusing {path} for {db.path}", flush=True)
                continue
        print(f"Building value index for {db.path}", flush=True)
        print(f"Saved {index.build(db, embed)}", flush=True)


if __name__ == "__main__":
    main()
