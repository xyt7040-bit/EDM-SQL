"""Read-only SQLite execution and schema grounding for the reference method."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp


def query_tree(sql: str) -> exp.Query | None:
    try:
        statements = [statement for statement in sqlglot.parse(sql, read="sqlite") if statement]
    except sqlglot.errors.SqlglotError:
        return None
    if len(statements) == 1 and isinstance(statements[0], exp.Query):
        return statements[0]
    return None


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class Execution:
    columns: tuple = ()
    rows: tuple = ()
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def record(self) -> dict:
        return {
            "success": self.success,
            "columns": self.columns,
            "row_count": len(self.rows),
            "sample_rows": self.rows[:8],
            "error": self.error,
        }

    def signature(self) -> tuple:
        return self.columns, self.rows


def authorize(action, name, argument, database, trigger):
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (argument or "").lower() in {
            "load_extension", "readfile", "writefile",
        } else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_OK if action in {
        sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE,
    } else sqlite3.SQLITE_DENY


class Database:
    def __init__(self, path: Path, timeout: float = 30.0):
        self.path = Path(path).resolve(strict=True)
        self.timeout = timeout
        self.cache: dict[str, Execution] = {}

    def _connect(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=self.timeout)
        deadline = time.monotonic() + self.timeout
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        return connection

    def execute(self, sql: str) -> Execution:
        if sql not in self.cache:
            if query_tree(sql) is None:
                self.cache[sql] = Execution(error="Not a single read-only query")
            else:
                try:
                    with closing(self._connect()) as connection:
                        connection.set_authorizer(authorize)
                        cursor = connection.execute(sql)
                        self.cache[sql] = Execution(
                            tuple(column[0] for column in cursor.description),
                            tuple(cursor.fetchall()),
                        )
                except sqlite3.Error as exc:
                    self.cache[sql] = Execution(error=str(exc))
        return self.cache[sql]

    def schema(self) -> str:
        rows = self.execute(
            "SELECT sql FROM sqlite_schema "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        )
        if not rows.success:
            raise RuntimeError(f"Could not read schema: {rows.error}")
        return "\n".join(row[0] for row in rows.rows)

    def text_value_batches(self, batch_size: int):
        with closing(self._connect()) as connection:
            # The SQL time budget should not include model inference between batches.
            deadline = time.monotonic() + self.timeout
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            tables = connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
            for (table,) in tables:
                deadline = time.monotonic() + self.timeout
                columns = connection.execute(
                    "PRAGMA table_info(" + quote_identifier(table) + ")"
                ).fetchall()
                for _, column, kind, *_ in columns:
                    column_type = kind.upper()
                    if column_type != "TEXT" and not column_type.startswith(("VARCHAR", "CHAR")):
                        continue
                    name = quote_identifier(column)
                    deadline = time.monotonic() + self.timeout
                    with closing(connection.execute(
                        f"SELECT DISTINCT {name} FROM {quote_identifier(table)} "
                        f"WHERE {name} IS NOT NULL AND length(CAST({name} AS TEXT)) <= 100 "
                        f"ORDER BY {name} COLLATE BINARY"
                    )) as cursor:
                        while rows := cursor.fetchmany(batch_size):
                            yield (table, column), [str(value) for (value,) in rows]
                            deadline = time.monotonic() + self.timeout

    def calibrate(self, sql: str) -> str:
        tree = query_tree(sql)
        if not isinstance(tree, exp.Select) or tree.find(exp.With):
            return sql
        tables = list(tree.find_all(exp.Table))
        if len(tables) != 1 or len(list(tree.find_all(exp.Select))) != 1:
            return sql
        table = tables[0]
        if table.db or table.catalog or not isinstance(table.this, exp.Identifier):
            return sql
        where = tree.args.get("where")
        if where is None:
            return sql
        changed = False
        with closing(self._connect()) as connection:
            for predicate in where.find_all(exp.EQ):
                column, value = predicate.left, predicate.right
                if not isinstance(column, exp.Column) or not isinstance(value, exp.Literal) or not value.is_string:
                    continue
                if column.table and column.table.lower() != table.alias_or_name.lower():
                    continue
                name = quote_identifier(column.name)
                try:
                    matches = connection.execute(
                        f"SELECT DISTINCT {name} COLLATE BINARY FROM {quote_identifier(table.name)} "
                        f"WHERE lower({name}) = lower(?) AND typeof({name}) = 'text' LIMIT 2",
                        (value.this,),
                    ).fetchall()
                except sqlite3.Error:
                    continue
                if len(matches) == 1 and matches[0][0] != value.this:
                    value.replace(exp.Literal.string(matches[0][0]))
                    changed = True
        return tree.sql(dialect="sqlite") if changed else sql
