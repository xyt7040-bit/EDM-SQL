import json
import os
import shutil
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import numpy as np

from edm_sql.database import Database
from edm_sql.pipeline import EDMSQL
from edm_sql.prompts import messages_for
from edm_sql.value_linking import ValueIndex, retrieve_values


CURRENT = "SELECT name FROM items WHERE id = 1"
ENHANCED = "SELECT name FROM items WHERE id = 2"
RECONSTRUCTED = "SELECT name FROM items WHERE id = 3"
BAD = "SELECT missing FROM items"
RETAIN = {"verdict": "retain", "defect": "No defect established", "evidence": "Candidate not supported"}
ACCEPT = {"verdict": "accept", "defect": "Incorrect binding", "evidence": "Question supports the change"}
CARD = {"applies_when": "A new binding is proposed", "risk": "Wrong binding", "guidance": "Check intent"}


class TestMethod(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / f"method-test-{uuid4().hex}.sqlite"
        self.addCleanup(path.unlink, missing_ok=True)
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(
                "CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
                "INSERT INTO items VALUES (1, 'Alpha'), (2, 'Beta'), (3, 'Gamma');"
            )
        self.db = Database(path)
        index_directory = path.with_suffix(".indices")
        index_directory.mkdir()
        self.value_index = ValueIndex(index_directory, "test-embedding")
        self.addCleanup(shutil.rmtree, index_directory)
        self.context = {"question": "List item names.", "schema": self.db.schema(), "augmented_schema": {}}

    def test_query_screening(self):
        def unexpected(*args):
            self.fail("Level 1 invoked the model")

        method = EDMSQL(unexpected)
        same = method.select_pair("SSM", self.context, self.db, CURRENT, CURRENT + ";")
        self.assertEqual((same.level, same.selected_sql, same.reason), (1, CURRENT, "redundant"))
        good = method.select_pair("SSM", self.context, self.db, BAD, CURRENT)
        self.assertEqual((good.level, good.verdict, good.selected_sql), (1, "accept", CURRENT))
        failed = method.select_pair("RSM", self.context, self.db, BAD, "SELECT also_missing FROM items")
        self.assertTrue(failed.unresolved)
        self.assertEqual(failed.reason, "neither_executes")
        missing = method.select_pair("RSM", self.context, self.db, CURRENT, "")
        self.assertEqual((missing.level, missing.selected_sql, missing.reason),
                         (1, CURRENT, "no_new_candidate"))
        self.assertFalse(self.db.execute("DELETE FROM items").success)
        self.assertFalse(self.db.execute("SELECT 1; SELECT 2").success)

    def test_order_columns_and_duplicates_are_preserved(self):
        left = self.db.execute("SELECT name FROM items ORDER BY id")
        reversed_order = self.db.execute("SELECT name FROM items ORDER BY id DESC")
        different_name = self.db.execute("SELECT name AS renamed FROM items ORDER BY id")
        duplicate = self.db.execute("SELECT 'Alpha' AS name UNION ALL SELECT 'Alpha'")
        self.assertNotEqual(left.signature(), reversed_order.signature())
        self.assertNotEqual(left.signature(), different_name.signature())
        self.assertEqual(duplicate.rows, (("Alpha",), ("Alpha",)))

    def test_level_two_rejects_only_explicit_violations(self):
        calls = []

        def complete(role, payload):
            calls.append((role, payload))
            return ACCEPT

        method = EDMSQL(complete)
        new = "SELECT name, id FROM items WHERE id = 1"
        strict = {**self.context, "question": "Return exactly one column."}
        rejected = method.select_pair("SSM", strict, self.db, CURRENT, new)
        self.assertEqual((rejected.level, rejected.selected_sql), (2, CURRENT))
        self.assertFalse(calls)
        permitted = method.select_pair("SSM", self.context, self.db, CURRENT, new)
        self.assertEqual((permitted.level, permitted.selected_sql), (3, new))
        self.assertIn("outputs changed", calls[0][1]["warnings"])

    def test_level_two_counts_executed_columns_for_wildcards(self):
        def unexpected(*args):
            self.fail("A verified output-shape violation reached adjudication")

        context = {**self.context, "question": "Return exactly one output column."}
        decision = EDMSQL(unexpected).select_pair(
            "SSM", context, self.db, CURRENT, "SELECT * FROM items WHERE id = 1",
        )
        self.assertEqual((decision.level, decision.selected_sql), (2, CURRENT))
        self.assertEqual(decision.evidence["difference"]["result_columns"],
                         {"current": ["name"], "new": ["id", "name"]})

    def test_level_two_separates_nested_and_outer_changes(self):
        def complete(role, payload):
            self.assertEqual(role, "adjudicate")
            return ACCEPT

        context = {**self.context, "question": "List matching item names, including duplicates."}
        new = ("SELECT name FROM items WHERE id IN "
               "(SELECT DISTINCT id FROM items WHERE id = 1)")
        decision = EDMSQL(complete).select_pair("RSM", context, self.db, "SELECT name FROM items", new)
        self.assertEqual((decision.level, decision.selected_sql), (3, new))
        self.assertIn("distinct changed in the nested query", decision.evidence["warnings"])
        self.assertNotIn("distinct changed in the outer query", decision.evidence["warnings"])
        filters = decision.evidence["difference"]["filters"]["scope_changes"]
        self.assertEqual(set(filters), {"outer", "nested"})

    def test_level_two_does_not_read_quoted_text_as_an_output_instruction(self):
        context = {**self.context, "question": 'Display the message "Return exactly one column." with its ID.'}
        new = "SELECT * FROM items WHERE id = 1"
        decision = EDMSQL(lambda role, payload: ACCEPT).select_pair("RSM", context, self.db, CURRENT, new)
        self.assertEqual((decision.level, decision.selected_sql), (3, new))

    def test_level_two_warns_when_current_and_new_already_violate_output_count(self):
        context = {**self.context, "question": "Return exactly one column."}
        decision = EDMSQL(lambda role, payload: RETAIN).select_pair(
            "SSM", context, self.db, "SELECT id, name FROM items", "SELECT id, name, 0 FROM items",
        )
        self.assertEqual(decision.level, 3)
        self.assertIn("result columns changed", decision.evidence["warnings"])

    def test_level_two_rejects_explicit_outer_query_violations(self):
        method = EDMSQL(lambda *args: self.fail("A verified violation reached adjudication"))
        cases = [
            ("SSM", "Count all rows, including duplicates.", "SELECT COUNT(*) FROM items",
             "SELECT COUNT(DISTINCT LENGTH(name)) FROM items", "changed_row_count"),
            ("RSM", "List names. Do not filter rows.", "SELECT name FROM items",
             "SELECT name FROM items WHERE id > 1", "added_filter"),
            ("RSM", "List name lengths. Do not group rows.", "SELECT LENGTH(name) FROM items",
             "SELECT LENGTH(name) FROM items GROUP BY LENGTH(name)", "added_aggregation"),
            ("RSM", "List item IDs. Do not aggregate.", "SELECT id FROM items",
             "SELECT SUM(id) FROM items", "added_aggregation"),
            ("RSM", "List name lengths. Preserve duplicate rows.", "SELECT LENGTH(name) FROM items",
             "SELECT DISTINCT LENGTH(name) FROM items", "added_deduplication"),
            ("RSM", "Return the values as stored.", "SELECT name FROM items",
             "SELECT LOWER(name) AS name FROM items", "transformed_output"),
        ]
        for stage, question, current, new, rule in cases:
            with self.subTest(rule=rule):
                decision = method.select_pair(stage, {**self.context, "question": question}, self.db, current, new)
                self.assertEqual((decision.level, decision.reason, decision.selected_sql), (2, rule, current))
                self.assertEqual(decision.evidence["violation"]["scope"], "outer")
                self.assertTrue(decision.evidence["violation"]["request"])
                if rule == "changed_row_count":
                    self.assertEqual(decision.evidence["violation"]["current_count"], 3)
                    self.assertEqual(decision.evidence["violation"]["candidate_count"], 2)
                if rule == "transformed_output":
                    self.assertEqual(decision.evidence["violation"]["output_column"], 1)

    def test_level_two_warns_when_count_or_output_values_do_not_change(self):
        method = EDMSQL(lambda role, payload: ACCEPT)
        cases = [
            ("SSM", "Count all rows, including duplicates.", "SELECT COUNT(*) FROM items",
             "SELECT COUNT(DISTINCT id) FROM items", "aggregates"),
            ("RSM", "Return the values as stored.", "SELECT name FROM items",
             "SELECT CAST(name AS TEXT) AS label FROM items", "transforms"),
            ("RSM", "Return the values as stored.", "SELECT * FROM items",
             "SELECT id, LOWER(name) FROM items", "transforms"),
            ("RSM", "Return the values as stored.", "SELECT *, name FROM items",
             "SELECT *, LOWER(name) FROM items", "transforms"),
        ]
        for stage, question, current, new, warning in cases:
            with self.subTest(candidate=new):
                decision = method.select_pair(stage, {**self.context, "question": question}, self.db, current, new)
                self.assertEqual((decision.level, decision.selected_sql), (3, new))
                self.assertNotIn("violation", decision.evidence)
                self.assertTrue(any(warning in item for item in decision.evidence["warnings"]))

    def test_level_two_warns_on_unconstrained_changes_or_unclear_scope(self):
        method = EDMSQL(lambda role, payload: ACCEPT)
        cases = [
            ("Count the items.", "SELECT COUNT(*) FROM items",
             "SELECT COUNT(DISTINCT LENGTH(name)) FROM items", "aggregates"),
            ("List the matching names.", "SELECT name FROM items",
             "SELECT name FROM items WHERE id > 1", "filters"),
            ("Display the message 'Do not filter rows.'", "SELECT name FROM items",
             "SELECT name FROM items WHERE id > 1", "filters"),
            ("List names. Do not group rows.", "SELECT name FROM items",
             "SELECT name FROM items WHERE id < (SELECT MAX(id) FROM items)", "aggregates"),
            ("Do not group rows.", "SELECT id FROM items", "SELECT SUM(id) FROM items", "aggregates"),
            ("Preserve duplicate rows.", "SELECT LENGTH(name) FROM items",
             "SELECT DISTINCT LENGTH(name) FROM items WHERE id > 1", "distinct"),
            ("Return values as stored.", "SELECT name FROM items", "SELECT UPPER(id) FROM items", "transforms"),
            ("Return exactly one column or two if necessary.", CURRENT,
             "SELECT name, id FROM items WHERE id = 1", "result columns"),
        ]
        for question, current, new, warning in cases:
            with self.subTest(question=question):
                decision = method.select_pair("RSM", {**self.context, "question": question}, self.db, current, new)
                self.assertEqual((decision.level, decision.selected_sql), (3, new))
                self.assertTrue(any(warning in item for item in decision.evidence["warnings"]))

    def test_memory_updates_after_both_stages_and_only_from_selected_feedback(self):
        calls = []
        method = None

        def complete(role, payload):
            calls.append((role, json.loads(json.dumps(payload))))
            return {
                "adjudicate": RETAIN, "decompose": {"steps": [{"description": "Get names", "sql": CURRENT}]},
                "check_step": {"pass": True}, "synthesize": {"sql": RECONSTRUCTED},
                "summarize": {"card": CARD}, "update": {"pool": [CARD]},
            }[role]

        def feedback(stage, sql):
            calls.append(("verdict", (stage, sql)))
            return True if stage == "SSM" else None

        method = EDMSQL(complete)
        question_evidence = "Use the case-sensitive item labels."
        result = method.answer("List item names.", self.db, initial_sql=CURRENT, schema_sql=ENHANCED,
                               feedback=feedback, generation_inputs={"evidence": question_evidence})
        self.assertEqual(result["delivered_sql"], CURRENT)
        self.assertEqual([role for role, _ in calls], [
            "adjudicate", "decompose", "check_step", "synthesize", "adjudicate",
            "verdict", "summarize", "update", "verdict",
        ])
        self.assertEqual(calls[0][1]["memory_pool"], [])
        self.assertEqual(calls[4][1]["memory_pool"], [])
        self.assertEqual(calls[6][1]["selected_sql"], CURRENT)
        self.assertTrue(calls[6][1]["outcome_verdict"])
        self.assertEqual(calls[6][1]["question_evidence"], question_evidence)
        self.assertEqual(calls[6][1]["evidence"], json.loads(json.dumps(result["ssm"]["evidence"])))
        prompt = messages_for("summarize", calls[6][1])[1]["content"]
        self.assertIn("BIRD evidence supplied with the question:\n" + question_evidence, prompt)
        self.assertIn("Candidate comparison and execution evidence:", prompt)
        self.assertNotIn("gold_sql", calls[6][1])
        self.assertEqual(method.memory, {"SSM": [CARD], "RSM": []})
        calls.clear()
        method.answer("List item names.", self.db, initial_sql=CURRENT, schema_sql=ENHANCED)
        self.assertEqual(calls[0][1]["memory_pool"], [CARD])
        self.assertEqual(calls[4][1]["memory_pool"], [])

    def test_adjudicator_receives_entire_pool_and_overflow_keeps_it(self):
        pool = [{**CARD, "guidance": str(index)} for index in range(7)]
        called = []

        def complete(role, payload):
            called.append((role, payload))
            return {
                "adjudicate": RETAIN, "summarize": {"card": CARD}, "update": {"pool": [CARD] * 11},
            }[role]

        method = EDMSQL(complete)
        method.memory["SSM"] = pool
        decision = method.select_pair("SSM", self.context, self.db, CURRENT, ENHANCED)
        method.close_question(self.context["question"], (decision,), lambda stage, sql: False)
        self.assertEqual(called[0][1]["memory_pool"], pool)
        self.assertEqual(called[2][1]["pool"], pool)
        self.assertEqual(method.memory["SSM"], pool)

    def test_no_memory_keeps_adjudication_but_skips_updates(self):
        called = []

        def complete(role, payload):
            called.append(role)
            self.assertEqual(payload["memory_pool"], [])
            return RETAIN

        def feedback(*args):
            self.fail("The memory-free control requested feedback")

        method = EDMSQL(complete, enable_memory=False)
        method.memory["SSM"] = [CARD]
        decision = method.select_pair("SSM", self.context, self.db, CURRENT, ENHANCED)
        method.close_question(self.context["question"], (decision,), feedback)
        self.assertEqual((decision.level, decision.selected_sql), (3, CURRENT))
        self.assertEqual(called, ["adjudicate"])

    def test_reconstruction_restarts_at_first_step_and_materializes_clauses(self):
        checked = []

        def complete(role, payload):
            if role == "decompose":
                return {"steps": [{"description": "Read names", "sql": "SELECT name FROM items"},
                                  {"description": "Filter rows", "sql": "WHERE unknown = 1"}]}
            if role == "check_step":
                checked.append((payload["index"], payload["execution_records"][-1]["sql"]))
                return {"pass": payload["execution_records"][-1]["success"]}
            if role == "revise_steps":
                return {"steps": [payload["steps"][0], {"description": "Filter rows", "sql": "WHERE id = 1"}]}
            return {"sql": payload["steps"][0]["sql"] + " " + payload["steps"][1]["sql"]}

        self.assertEqual(EDMSQL(complete).reconstruct(self.context, self.db, CURRENT), CURRENT)
        self.assertEqual([index for index, _ in checked], [0, 1, 0, 1])
        self.assertEqual(checked[-1][1], CURRENT)

    def test_five_round_budget_still_synthesizes_latest_steps(self):
        roles = []

        def complete(role, payload):
            roles.append(role)
            if role == "decompose":
                return {"steps": [{"description": "Check item", "sql": BAD}]}
            if role == "check_step":
                return {"pass": False, "diagnostic": "Unresolved"}
            if role == "revise_steps":
                return {"steps": payload["steps"]}
            return {"sql": CURRENT}

        self.assertEqual(EDMSQL(complete).reconstruct(self.context, self.db, CURRENT), CURRENT)
        self.assertEqual(roles.count("check_step"), 5)
        self.assertEqual(roles.count("revise_steps"), 5)
        self.assertEqual(roles[-1], "synthesize")

    def test_optimizer_has_one_repair_and_ordered_cached_fallback(self):
        roles = []

        def complete(role, payload):
            roles.append(role)
            return {"sql": "DROP TABLE items"}

        self.db.execute(CURRENT)
        self.db.execute(ENHANCED)
        deliver = EDMSQL(complete).deliver
        delivery = deliver(self.context, self.db, BAD, ENHANCED, CURRENT, RECONSTRUCTED)
        self.assertEqual((delivery.status, delivery.source, delivery.sql), ("resolved", "ssm", ENHANCED))
        self.assertEqual(roles, ["execution_repair"])
        self.assertEqual(deliver(self.context, self.db, BAD, "", "", "").status, "unresolved")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM items").rows, ((3,),))

    def test_optimizer_calibrates_unambiguous_text_values(self):
        sql = "SELECT name FROM items WHERE name = 'alpha'"

        def unexpected(*args):
            self.fail("Unique literal calibration invoked an LLM")

        delivery = EDMSQL(unexpected).deliver(self.context, self.db, sql, "", "", "")
        self.assertEqual((delivery.status, delivery.execution.rows), ("resolved", (("Alpha",),)))

    def test_optimizer_preserves_successful_empty_results(self):
        def unexpected(*args):
            self.fail("An empty result alone must not trigger a repair")

        for question, sql in (
            ("Return the name of item 99.", "SELECT name FROM items WHERE id = 99"),
            ("Return names after excluding all stored items.",
             "SELECT name FROM items EXCEPT SELECT name FROM items"),
        ):
            with self.subTest(sql=sql):
                context = {**self.context, "question": question}
                delivery = EDMSQL(unexpected).deliver(context, self.db, sql, "", "", "")
                self.assertEqual((delivery.sql, delivery.source, delivery.status), (sql, "optimized", "resolved"))
                self.assertTrue(delivery.execution.success)
                self.assertEqual(delivery.execution.rows, ())
                self.assertIsNone(delivery.risk)

    def test_optimizer_does_not_flag_a_count_of_zero(self):
        def unexpected(*args):
            self.fail("A scalar count of zero triggered an empty-result review")

        sql = "SELECT COUNT(*) FROM items WHERE id = 99"
        delivery = EDMSQL(unexpected).deliver(self.context, self.db, sql, "", "", "")
        self.assertEqual(delivery.execution.rows, ((0,),))
        self.assertIsNone(delivery.risk)

    def test_schema_augmentation_passes_three_grounding_signals(self):
        seen = []
        inputs = {"few_shot": "Worked example.", "evidence": "Use the item name.",
                  "intent_profile": "Lookup", "intent_soft_guidance": "Filter by name."}

        def complete(role, payload):
            if role == "generate_sql":
                for name, value in inputs.items():
                    self.assertEqual(payload[name], value)
                if "augmented_schema" in payload:
                    seen.append(payload["augmented_schema"])
                return {"SQL": CURRENT}
            if role == "direct_schema_links":
                self.assertEqual(payload["question_evidence"], inputs["evidence"])
                prompt = messages_for(role, payload)[1]["content"]
                self.assertIn("BIRD evidence supplied with the question:\n" + inputs["evidence"], prompt)
                return {"links": [{"table": "items", "column": "name"}]}
            if role == "value_mentions":
                self.assertEqual(payload["evidence"], inputs["evidence"])
                return {"mentions": ["first letter", "first letter"]}
            return {"steps": []}

        embedded = []

        def embed(texts):
            embedded.append(texts)
            vectors = {"first letter": [1., 0.], "Alpha": [1., 0.],
                       "Beta": [0., 1.], "Gamma": [-1., 0.]}
            return np.array([vectors[text] for text in texts])

        self.value_index.build(self.db, embed)
        self.assertEqual(embedded, [["Alpha", "Beta", "Gamma"]])
        embedded.clear()
        method = EDMSQL(complete, embed=embed, value_index=self.value_index)
        result = method.answer("Show the first letter.", self.db, generation_inputs=inputs)
        self.assertEqual((result["initial_sql"], result["schema_sql"]), (CURRENT, CURRENT))
        self.assertEqual(embedded, [["first letter"]])
        self.assertEqual(seen, [{
            "direct": [{"table": "items", "column": "name"}],
            "values": [{"table": "items", "column": "name", "value": value}
                       for value in ("Alpha", "Beta", "Gamma")],
            "sql_anchored": {"tables": ["items"], "columns": ["id", "name"]},
        }])

    def test_embedding_retrieval_keeps_top_values_across_batches(self):
        class BatchedDatabase:
            path = self.db.path

            def text_value_batches(self, batch_size):
                yield ("items", "name"), ["A", "B", "C", "D", "E"]
                yield ("items", "name"), ["F", "G"]
                yield ("items", "category"), ["H"]
                yield ("items", "numeric_text"), ["123", "456.7"]
                yield ("items", "uuid_text"), ["11111111-2222-3333-4444-555555555555"]

        db = BatchedDatabase()
        embed = lambda values: np.array([[float(ord(value))] for value in values])
        path = self.value_index.build(db, embed)
        links = retrieve_values(path, ["A"], embed)
        self.assertEqual([item["value"] for item in links], ["H", "G", "F", "E", "D", "C"])

    def test_value_mentions_fall_back_only_for_unusable_output(self):
        self.value_index.build(self.db, lambda values: np.ones((len(values), 2)))
        for response, expected in [
            ({}, [["Find", "Alpha", "Use", "names."]]),
            ({"mentions": [" ", 7]}, [["Find", "Alpha", "Use", "names."]]),
            ({"mentions": []}, []),
        ]:
            with self.subTest(response=response):
                embedded, value_links = [], []

                def embed(terms):
                    embedded.append(terms)
                    return np.ones((len(terms), 2))

                def complete(role, payload):
                    if role == "value_mentions":
                        return response
                    if role == "generate_sql":
                        value_links.extend(payload["augmented_schema"]["values"])
                        return {"SQL": CURRENT}
                    return {}

                method = EDMSQL(complete, embed=embed, value_index=self.value_index)
                method.answer("Find Alpha", self.db, initial_sql=CURRENT,
                              generation_inputs={"evidence": "Use Alpha names."})
                self.assertEqual(embedded, expected)
                self.assertEqual(bool(value_links), bool(expected))

    def test_question_evidence_reaches_later_roles_separately_from_execution_records(self):
        question_evidence = "Use the supplied case-sensitive value mapping."
        wildcard_sql = "SELECT * FROM items WHERE id = 3"
        steps = [{"description": "Find the item name", "sql": ENHANCED}]
        calls = []

        def complete(role, payload):
            self.assertEqual(payload["question_evidence"], question_evidence)
            prompt = messages_for(role, payload)[1]["content"]
            self.assertIn("BIRD evidence supplied with the question:\n" + question_evidence, prompt)
            calls.append(role)
            if role == "adjudicate":
                self.assertIn("Execution record for the current SQL:", prompt)
                self.assertNotIn("evidence", payload)
                return ACCEPT
            if role == "decompose":
                return {"steps": [] if payload["current_sql"] == BAD else steps}
            if role == "check_step":
                return {"pass": calls.count(role) > 1, "diagnostic": "Check the value mapping"}
            if role == "revise_steps":
                return {"steps": steps}
            if role == "synthesize":
                return {"sql": wildcard_sql}
            if role == "guarded_repair":
                return {"sql": None}
            if role == "execution_repair":
                self.assertIn("Execution error:", prompt)
                return {"sql": CURRENT}
            self.fail(f"Unexpected role {role}")

        method = EDMSQL(complete)
        method.answer("Find the matching item.", self.db, initial_sql=CURRENT, schema_sql=ENHANCED,
                      generation_inputs={"evidence": question_evidence})
        method.answer("Find the matching item.", self.db, initial_sql=BAD, schema_sql=BAD,
                      generation_inputs={"evidence": question_evidence})
        self.assertEqual(calls, ["adjudicate", "decompose", "check_step", "revise_steps",
                                 "check_step", "synthesize", "adjudicate", "guarded_repair",
                                 "decompose", "execution_repair"])

    def test_missing_value_index_stops_before_model_calls(self):
        def unexpected(*args):
            self.fail("A missing index must not trigger a model call")

        method = EDMSQL(unexpected, embed=unexpected, value_index=self.value_index)
        with self.assertRaisesRegex(FileNotFoundError, "build_value_index.py"):
            method.answer("Show item names.", self.db)

    def test_offline_value_index_rejects_wrong_model_or_stale_database(self):
        self.value_index.build(self.db, lambda values: np.ones((len(values), 2)))
        self.assertTrue((self.value_index.require(self.db) / "manifest.json").is_file())
        other_model = ValueIndex(self.value_index.root, "other-model")
        with self.assertRaisesRegex(ValueError, "incompatible or stale"):
            other_model.require(self.db)
        stat = self.db.path.stat()
        os.utime(self.db.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        with self.assertRaisesRegex(ValueError, "incompatible or stale"):
            self.value_index.require(self.db)

    def test_index_without_edm_sql_metadata_is_rejected_and_preserved(self):
        embed = lambda values: np.ones((len(values), 2))
        path = self.value_index.build(self.db, embed)
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["edm_sql"]
        contents = json.dumps(manifest)
        manifest_path.write_text(contents, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lacks EDM-SQL metadata"):
            self.value_index.require(self.db)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.value_index.build(self.db, embed)
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), contents)

    def test_precomputed_schema_context_does_not_require_embeddings(self):
        def complete(role, payload):
            if role == "generate_sql":
                self.assertEqual(payload["augmented_schema"], {"values": []})
                return {"SQL": CURRENT}
            self.assertEqual(role, "decompose")
            return {"steps": []}

        method = EDMSQL(complete)
        with self.assertRaisesRegex(ValueError, "embedding model"):
            method.answer("Show item names.", self.db, initial_sql=CURRENT)
        self.assertEqual(method.answer("Show item names.", self.db, initial_sql=CURRENT,
                                       schema_context={"values": []})["schema_sql"], CURRENT)

    def test_generation_prompt_renders_inputs_and_json_contract(self):
        payload = {"question": "Find {Alpha}.", "schema": self.db.schema(),
                   "few_shot": [{"question": "Example?", "sql": CURRENT}],
                   "evidence": "Use names.", "intent_profile": "Lookup",
                   "intent_soft_guidance": "Filter by name."}
        system, user = messages_for("generate_sql", payload)
        self.assertEqual((system["role"], user["role"]), ("system", "user"))
        for value in ("Find {Alpha}.", "Example?", "Use names.", "Lookup", "Filter by name.",
                      '{"SQL":"SELECT..."}'):
            self.assertIn(value, user["content"])
        self.assertNotIn("augmented_section", user["content"])
        self.assertNotIn("linked_value", user["content"])
        payload["augmented_schema"] = {"values": ["linked_value"]}
        self.assertIn('{"values": ["linked_value"]}', messages_for("generate_sql", payload)[1]["content"])
        del payload["evidence"]
        self.assertNotIn("{evidence}", messages_for("generate_sql", payload)[1]["content"])

    def test_other_role_templates_render_inputs_and_output_contract(self):
        common = "question schema augmented_schema"
        contracts = {
            "direct_schema_links": ("question schema", {"links"}),
            "value_mentions": ("question evidence", {"mentions"}),
            "adjudicate": (common + " stage current_sql new_sql current_execution new_execution "
                           "difference warnings memory_pool", {"verdict", "defect", "evidence"}),
            "summarize": ("stage question current_sql new_sql selected_sql selection evidence "
                          "rationale outcome_verdict", {"card"}),
            "update": ("stage card pool capacity", {"pool"}),
            "decompose": (common + " current_sql", {"steps"}),
            "check_step": (common + " steps index execution_records", {"pass", "diagnostic"}),
            "revise_steps": (common + " steps index execution diagnostic", {"steps"}),
            "synthesize": (common + " steps", {"sql"}),
            "guarded_repair": (common + " sql risks", {"sql"}),
            "execution_repair": (common + " sql error", {"sql"}),
        }
        for role, (inputs, keys) in contracts.items():
            with self.subTest(role=role):
                payload = {name: "VALUE_" + name.upper() for name in inputs.split()}
                system, user = messages_for(role, payload)
                self.assertEqual((system["role"], user["role"]), ("system", "user"))
                self.assertIn("EDM-SQL", system["content"])
                prompt = user["content"]
                self.assertNotIn("BIRD evidence supplied with the question", prompt)
                for value in payload.values():
                    self.assertIn(value, prompt)
                example, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{"):])
                self.assertEqual(set(example), keys)
        _, user = messages_for("summarize", {**dict.fromkeys(contracts["summarize"][0].split(), ""),
                                               "outcome_verdict": False})
        self.assertIn("(true = correct, false = incorrect): false", user["content"])

    def test_decomposition_and_revision_share_executable_step_requirements(self):
        context = {"question": "Find Alpha.", "schema": "items(id, name)"}
        steps = [{"description": "List the items", "sql": CURRENT}]
        decomposition = messages_for("decompose", {**context, "current_sql": CURRENT})[1]["content"]
        revision = messages_for("revise_steps", {
            **context, "steps": steps, "index": 0, "execution": {}, "diagnostic": "Invalid clause",
        })[1]["content"]
        for prompt in (decomposition, revision):
            self.assertIn("description match what its SQL actually executes", prompt)
            self.assertIn("step must be a complete query", prompt)
            self.assertIn("reordering or replacing any existing clauses", prompt)
            self.assertNotIn("{step_sql_requirements}", prompt)
        self.assertIn("it starts at Step 1", revision)

    def test_step_verifier_adds_evidence_checks_only_when_evidence_is_supplied(self):
        payload = {**self.context, "steps": [{"description": "List the items", "sql": CURRENT}],
                   "index": 0, "execution_records": []}
        generic = messages_for("check_step", payload)[1]["content"]
        evidence = "Use each category's total as the denominator."
        grounded = messages_for("check_step", {**payload, "question_evidence": evidence})[1]["content"]
        self.assertIn(evidence, grounded)
        for phrase in ("formula, denominator", "date format or granularity", "value mapping",
                       "unit or conversion", "aggregation grain"):
            self.assertIn(phrase, grounded)
            self.assertNotIn(phrase, generic)
        self.assertNotIn("{step_evidence_checks}", generic)

    def test_synthetic_decomposition_example_checks_both_counts(self):
        prompt = messages_for("decompose", {**self.context, "current_sql": CURRENT})[1]["content"]
        example, _, actual_input = prompt.partition("Now decompose the actual input")
        steps, _ = json.JSONDecoder().raw_decode(example.split("Example step list:\n", 1)[1])
        self.assertIn("Synthetic example", example)
        self.assertNotIn(CURRENT, example)
        self.assertIn(CURRENT, actual_input)
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.executescript("CREATE TABLE demo_members(member_id INTEGER PRIMARY KEY);"
                               "CREATE TABLE demo_events(member_id INTEGER);"
                               "INSERT INTO demo_members VALUES (1), (2), (3), (4);"
                               "INSERT INTO demo_events VALUES (1), (1), (2);")
            self.assertEqual([conn.execute(step["sql"]).fetchone()[0] for step in steps["steps"]],
                             [4, 2, 50.0])


if __name__ == "__main__":
    unittest.main()
