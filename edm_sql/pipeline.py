"""EDM-SQL reference workflow."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Optional

from sqlglot import exp

from .database import Database, Execution, query_tree
from .prompts import GENERATION_CONTEXT_FIELDS
from .value_linking import retrieve_values


STAGES = ("SSM", "RSM")
MEMORY_LIMIT = 10
VERIFICATION_ROUNDS = 5
CARD_FIELDS = ("applies_when", "risk", "guidance")
CLAUSE = re.compile(r"^(?:WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET|"
                    r"(?:(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL)\s+)*JOIN)\b", re.I)
SINGLE_COLUMN_REQUEST = re.compile(
    r"(?:^|[.!?;]\s*)(?:please\s+)?(?:return|show|output|select|report)\s+"
    r"(?:(?:only|exactly)\s+one|(?:(?:only|exactly)\s+)?a single)\s+"
    r"(?:output\s+)?column(?=\s*(?:[.!?;]|$))",
    re.I,
)
QUOTED_TEXT = re.compile(r"""\x60[^\x60]*\x60|"[^"]*"|(?<!\w)'[^'\n]*'(?!\w)""")
EXPLICIT_CONSTRAINTS = {
    "no_filter": r"(?:do not|don't) (?:filter (?:any )?rows|apply (?:any )?filters)",
    "no_aggregate": r"(?:do not|don't) aggregate(?: the (?:rows|results))?",
    "no_group": r"(?:do not|don't) group (?:the )?rows",
    "keep_duplicates": r"(?:(?:keep|preserve|include) (?:all )?duplicate rows|(?:do not|don't) deduplicate(?: the rows)?)",
    "count_duplicates": r"count all rows,? including duplicates",
    "as_stored": r"(?:return|show|output) (?:the )?values (?:exactly )?as stored",
}
SCOPED_RISKS = {
    "filters": (exp.Where, exp.Having),
    "aggregates": (exp.AggFunc,),
    "grouping": (exp.Group,),
    "distinct": (exp.Distinct,),
    "transforms": (exp.Cast, exp.Case, exp.Lower, exp.Upper),
}
Completion = Callable[[str, dict], Mapping]
Feedback = Callable[[str, str], Optional[bool]]


def text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def search_keywords(extracted: object, question: str, question_evidence: object) -> list[str]:
    if isinstance(extracted, list) and all(text(item) for item in extracted):
        keywords = [text(item) for item in extracted]
    else:
        keywords = question.split() + text(question_evidence).split()
    return list(dict.fromkeys(keywords))


def render(expression) -> str:
    return expression.sql(dialect="sqlite", comments=False, normalize=True)


def features(tree: exp.Query) -> dict:
    return {
        "outputs": [render(item) for item in tree.selects],
        "bindings": sorted({render(item) for item in tree.find_all(exp.Column)}),
        "tables": sorted({item.name for item in tree.find_all(exp.Table)}),
        "joins": [render(item) for item in tree.find_all(exp.Join)],
        **{key: [render(item) for item in tree.find_all(*kinds)] for key, kinds in SCOPED_RISKS.items()},
        "order": [render(item) for item in tree.find_all(exp.Order)],
        "limit": [render(item) for item in tree.find_all(exp.Limit, exp.Offset)],
    }


def scoped_expressions(tree: exp.Query, kinds: tuple) -> dict[str, list[str]]:
    scopes: dict[str, list[str]] = {"outer": [], "nested": []}
    for item in tree.find_all(*kinds):
        owner = item.find_ancestor(exp.Query)
        scope = "outer" if owner is tree else "nested"
        scopes[scope].append(render(item))
    return scopes


def explicit_constraint(request: str, key: str) -> str | None:
    # Limit hard checks to standalone directives, rather than guessing intent from keywords.
    match = re.search(r"(?:^|[.!?;]\s*)(?:please\s+)?(" + EXPLICIT_CONSTRAINTS[key]
                      + r")(?=\s*(?:[.!?;]|$))", request, re.I)
    return match.group(1) if match else None


def outer_aggregation(tree: exp.Select) -> bool:
    return any(
        item.find_ancestor(exp.Query) is tree and item.find_ancestor(exp.Window) is None
        for item in tree.find_all(exp.AggFunc)
    )


def same_except(current: exp.Select, new: exp.Select, clause: str) -> bool:
    left, right = current.copy(), new.copy()
    left.set(clause, None)
    right.set(clause, None)
    return render(left) == render(right)


def projection_expression(item):
    return item.this if isinstance(item, exp.Alias) else item


def scalar_count(execution: Execution) -> int | None:
    if len(execution.rows) == 1 and len(execution.rows[0]) == 1:
        value = execution.rows[0][0]
        if type(value) is int:
            return value
    return None


def hard_violation(stage, question, current, new, current_execution, new_execution):
    request = QUOTED_TEXT.sub(" ", question).strip()
    single_column = SINGLE_COLUMN_REQUEST.search(request)
    if single_column and len(current_execution.columns) == 1 and len(new_execution.columns) > 1:
        return {"rule": "explicit_single_output", "request": single_column.group().strip(),
                "scope": "outer", "feature": "result_columns"}
    if not isinstance(current, exp.Select) or not isinstance(new, exp.Select):
        return None

    def violation(rule, constraint, feature):
        cue = explicit_constraint(request, constraint)
        return {"rule": rule, "request": cue, "scope": "outer", "feature": feature} if cue else None

    old_items = [projection_expression(item) for item in current.selects]
    new_items = [projection_expression(item) for item in new.selects]
    if (len(old_items) == len(new_items) == 1
            and isinstance(old_items[0], exp.Count) and isinstance(old_items[0].this, exp.Star)
            and isinstance(new_items[0], exp.Count) and isinstance(new_items[0].this, exp.Distinct)
            and not current.args.get("group") and not new.args.get("group")
            and same_except(current, new, "expressions")):
        if matched := violation("changed_row_count", "count_duplicates", "aggregates"):
            old_count = scalar_count(current_execution)
            new_count = scalar_count(new_execution)
            if old_count is not None and new_count is not None and new_count < old_count:
                return {**matched, "current_count": old_count, "candidate_count": new_count}
    if stage != "RSM":
        return None
    if (not current.args.get("where") and new.args.get("where")
            and len(new_execution.rows) < len(current_execution.rows)
            and same_except(current, new, "where")):
        if matched := violation("added_filter", "no_filter", "filters"):
            return matched
    if not current.args.get("group") and new.args.get("group"):
        if matched := violation("added_aggregation", "no_group", "grouping"):
            return matched
    if not outer_aggregation(current) and outer_aggregation(new):
        if matched := violation("added_aggregation", "no_aggregate", "aggregates"):
            return matched
    if (not current.args.get("distinct") and new.args.get("distinct")
            and len(new_execution.rows) < len(current_execution.rows)
            and same_except(current, new, "distinct")):
        if matched := violation("added_deduplication", "keep_duplicates", "distinct"):
            return matched
    matched = violation("transformed_output", "as_stored", "transforms")
    if (matched and len(old_items) == len(new_items) == len(current_execution.columns) == len(new_execution.columns)
            and len(current_execution.rows) == len(new_execution.rows)
            and same_except(current, new, "expressions")):
        for index, (before, after) in enumerate(zip(old_items, new_items)):
            if (isinstance(before, exp.Column) and isinstance(after, (exp.Cast, exp.Lower, exp.Upper))
                    and isinstance(after.this, exp.Column) and render(before) == render(after.this)):
                # A change in row order or column label is not a change in the values.
                old_values = Counter(row[index] for row in current_execution.rows)
                new_values = Counter(row[index] for row in new_execution.rows)
                if old_values != new_values:
                    return {**matched, "output_column": index + 1}
    return None


def compare(stage: str, question: str, current: exp.Query, new: exp.Query,
            current_execution: Execution, new_execution: Execution) -> tuple[dict, dict | None, list[str]]:
    old, proposed = features(current), features(new)
    old["result_columns"] = list(current_execution.columns)
    proposed["result_columns"] = list(new_execution.columns)
    difference = {key: {"current": old[key], "new": proposed[key]}
                  for key in old if old[key] != proposed[key]}
    relevant = {
        "SSM": ("outputs", "result_columns", "bindings", "tables", "aggregates", "grouping", "distinct", "joins", "filters"),
        "RSM": ("outputs", "result_columns", "joins", "filters", "aggregates", "grouping",
                "distinct", "transforms", "order", "limit"),
    }[stage]
    warnings = []
    for key in relevant:
        if key in SCOPED_RISKS:
            old_scopes = scoped_expressions(current, SCOPED_RISKS[key])
            new_scopes = scoped_expressions(new, SCOPED_RISKS[key])
            changed = {scope: {"current": before, "new": new_scopes[scope]}
                       for scope, before in old_scopes.items() if before != new_scopes[scope]}
            if changed:
                delta = difference.setdefault(key, {"current": old[key], "new": proposed[key]})
                delta["scope_changes"] = changed
                warnings.extend(f"{key} changed in the {scope} query" for scope in changed)
                continue
        if key in difference:
            warnings.append(key.replace("_", " ") + " changed")
    return difference, hard_violation(stage, question, current, new, current_execution, new_execution), warnings


def valid_card(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    card = {field: text(value.get(field)) for field in CARD_FIELDS}
    return card if all(card.values()) else None


def valid_steps(values: object) -> list[dict] | None:
    if not isinstance(values, list) or not values:
        return None
    steps = []
    for step in values:
        if not isinstance(step, Mapping):
            return None
        description, sql = text(step.get("description")), text(step.get("sql"))
        if not description or not sql:
            return None
        steps.append({"description": description, "sql": sql})
    return steps


@dataclass(frozen=True)
class Decision:
    stage: str
    level: int
    verdict: str
    reason: str
    selected_sql: str
    current_sql: str
    new_sql: str
    unresolved: bool = False
    evidence: dict | None = None
    rationale: dict | None = None


@dataclass(frozen=True)
class Delivery:
    sql: str
    source: str
    status: str
    execution: Execution
    risk: str | None = None


class EDMSQL:
    def __init__(self, complete: Completion, *, enable_memory: bool = True, embed=None, value_index=None):
        self.complete = complete
        self.enable_memory = enable_memory
        self.embed = embed
        self.value_index = value_index
        self.memory: dict[str, list[dict[str, str]]] = {stage: [] for stage in STAGES}

    def ask(self, role: str, payload: dict) -> Mapping:
        result = self.complete(role, payload)
        return result if isinstance(result, Mapping) else {}

    def select_pair(self, stage: str, context: dict, db: Database, current: str, new: str) -> Decision:
        def decide(level, reason, *, accept=False, unresolved=False, evidence=None, rationale=None):
            return Decision(stage, level, "accept" if accept else "retain", reason,
                            new if accept else current, current, new, unresolved, evidence, rationale)

        if not new:
            return decide(1, "no_new_candidate", unresolved=not db.execute(current).success)
        old_execution, new_execution = db.execute(current), db.execute(new)
        if not old_execution.success:
            if new_execution.success:
                return decide(1, "only_new_executes", accept=True)
            return decide(1, "neither_executes", unresolved=True)
        if not new_execution.success:
            return decide(1, "only_current_executes")
        old_tree, new_tree = query_tree(current), query_tree(new)
        if render(old_tree) == render(new_tree) or old_execution.signature() == new_execution.signature():
            return decide(1, "redundant")
        difference, violation, warnings = compare(
            stage, context["question"], old_tree, new_tree, old_execution, new_execution,
        )
        if violation:
            return decide(2, violation["rule"], evidence={"difference": difference, "warnings": warnings,
                                                          "violation": violation})
        evidence = {
            "current_execution": old_execution.record(), "new_execution": new_execution.record(),
            "difference": difference, "warnings": warnings,
        }
        response = self.ask("adjudicate", {
            **context, **evidence, "stage": stage, "current_sql": current, "new_sql": new,
            "memory_pool": list(self.memory[stage]) if self.enable_memory else [],
        })
        rationale = {key: text(response.get(key)) for key in ("verdict", "defect", "evidence")}
        accept = all(rationale.values()) and rationale["verdict"] == "accept"
        return decide(3, "semantic_adjudication", accept=accept, evidence=evidence, rationale=rationale)

    def close_question(self, question: str, decisions: tuple[Decision, ...],
                       feedback: Feedback | None, *, question_evidence: object = None) -> None:
        if not self.enable_memory or feedback is None:
            return
        for decision in decisions:
            if decision.level != 3:
                continue
            outcome = feedback(decision.stage, decision.selected_sql)
            if not isinstance(outcome, bool):
                continue
            stage = decision.stage
            proposed = self.ask("summarize", {
                "stage": stage, "question": question, "current_sql": decision.current_sql,
                "new_sql": decision.new_sql, "selected_sql": decision.selected_sql,
                "selection": decision.verdict, "evidence": decision.evidence,
                "question_evidence": question_evidence, "rationale": decision.rationale,
                "outcome_verdict": outcome,
            })
            card = valid_card(proposed.get("card"))
            if card is None:
                continue
            proposal = self.ask("update", {
                "stage": stage, "card": card, "pool": list(self.memory[stage]),
                "capacity": MEMORY_LIMIT,
            }).get("pool")
            if isinstance(proposal, list) and len(proposal) <= MEMORY_LIMIT:
                cards = [valid_card(entry) for entry in proposal]
                if all(entry is not None for entry in cards):
                    self.memory[stage] = cards

    def reconstruct(self, context: dict, db: Database, sql: str) -> str:
        if not sql:
            return ""
        steps = valid_steps(self.ask("decompose", {**context, "current_sql": sql}).get("steps"))
        if steps is None:
            return ""
        for _ in range(VERIFICATION_ROUNDS):
            records = []
            previous = ""
            for index, step in enumerate(steps):
                fragment = step["sql"]
                materialized = previous.rstrip(";") + " " + fragment if previous and CLAUSE.match(fragment) else fragment
                execution = db.execute(materialized)
                records.append({"sql": materialized, **execution.record()})
                verdict = self.ask("check_step", {
                    **context, "steps": steps, "index": index, "execution_records": records,
                })
                if verdict.get("pass") is True and execution.success:
                    previous = materialized
                    continue
                steps = valid_steps(self.ask("revise_steps", {
                    **context, "steps": steps, "index": index, "execution": records[-1],
                    "diagnostic": text(verdict.get("diagnostic")) or execution.error,
                }).get("steps"))
                if steps is None:
                    return ""
                break
            else:
                break
        final = text(self.ask("synthesize", {**context, "steps": steps}).get("sql"))
        return final if query_tree(final) is not None else ""

    def deliver(self, context: dict, db: Database, current: str, after_ssm: str,
                initial: str, enhanced: str) -> Delivery:
        latest = db.calibrate(current)
        old_tree = query_tree(latest)
        if old_tree is not None:
            used = features(old_tree)
            risks = []
            if any(output == "*" or output.endswith(".*") for output in used["outputs"]):
                risks.append("wildcard projection")
            if any(not join.args.get("on") and not join.args.get("using") for join in old_tree.find_all(exp.Join)):
                risks.append("join without an explicit condition")
            if used["grouping"] and not used["aggregates"]:
                risks.append("grouping without an aggregate")
            if risks:
                repaired = text(self.ask("guarded_repair", {
                    **context, "sql": latest, "risks": risks,
                }).get("sql"))
                if repaired and db.execute(repaired).success:
                    latest = repaired
        execution = db.execute(latest)
        if execution.success:
            return Delivery(latest, "optimized", "resolved", execution)
        if latest:
            repaired = text(self.ask("execution_repair", {
                **context, "sql": latest, "error": execution.error,
            }).get("sql"))
            if repaired and (result := db.execute(repaired)).success:
                return Delivery(repaired, "execution_repair", "resolved", result)
        for source, candidate in (("pre_optimizer", current), ("ssm", after_ssm),
                                  ("initial", initial), ("schema_enhanced", enhanced)):
            # Only cached execution records are eligible as delivery checkpoints.
            result = db.cache.get(candidate)
            if result is not None and result.success:
                return Delivery(candidate, source, "resolved", result)
        return Delivery(current, "pre_optimizer", "unresolved",
                        db.cache.get(current, Execution(error="No executable checkpoint")))

    def answer(self, question: str, db: Database, *, schema: str | None = None,
               initial_sql: str | None = None, schema_sql: str | None = None,
               generation_inputs: Mapping | None = None,
               schema_context: Mapping | None = None, feedback: Feedback | None = None) -> dict:
        needs_linking = schema_context is None and schema_sql is None
        if needs_linking and (self.embed is None or self.value_index is None):
            raise ValueError("Configure an embedding model and an offline value index, or provide precomputed augmented_schema")
        index_path = self.value_index.require(db) if needs_linking else None
        db.cache.clear()
        generation_fields = {key: (generation_inputs or {}).get(key) for key in GENERATION_CONTEXT_FIELDS}
        question_evidence = generation_fields["evidence"]
        context = {"question": question, "schema": db.schema() if schema is None else schema,
                   "question_evidence": question_evidence}

        def generate():
            response = self.ask("generate_sql", {**context, **generation_fields})
            return text(response.get("SQL")) or text(response.get("sql"))

        if initial_sql is None:
            initial_sql = generate()
        if schema_context is None:
            tree = query_tree(initial_sql)
            direct, values = [], []
            if needs_linking:
                direct = self.ask("direct_schema_links", context).get("links") or []
                extracted = self.ask("value_mentions", {
                    "question": question, "evidence": question_evidence,
                }).get("mentions")
                keywords = search_keywords(extracted, question, question_evidence)
                values = retrieve_values(index_path, keywords, self.embed)
            schema_context = {
                "direct": direct,
                "values": values,
                "sql_anchored": ({
                    "tables": features(tree)["tables"], "columns": features(tree)["bindings"],
                } if tree is not None else {}),
            }
        context["augmented_schema"] = dict(schema_context)
        if schema_sql is None:
            schema_sql = generate()
        ssm = self.select_pair("SSM", context, db, initial_sql, schema_sql)
        reconstructed = self.reconstruct(context, db, ssm.selected_sql)
        rsm = self.select_pair("RSM", context, db, ssm.selected_sql, reconstructed)
        delivery = self.deliver(context, db, rsm.selected_sql, ssm.selected_sql, initial_sql, schema_sql)
        self.close_question(question, (ssm, rsm), feedback, question_evidence=question_evidence)
        return {
            "initial_sql": initial_sql, "schema_sql": schema_sql, "reconstructed_sql": reconstructed,
            "ssm": asdict(ssm), "rsm": asdict(rsm), "delivered_sql": delivery.sql,
            "delivery_source": delivery.source, "status": delivery.status,
            "delivery_risk": delivery.risk,
            "result": asdict(delivery.execution),
        }
