"""Prompt templates and JSON contracts for EDM-SQL roles."""

import json


GENERATION_CONTEXT_FIELDS = ("few_shot", "evidence", "intent_profile", "intent_soft_guidance")

STEP_SQL_REQUIREMENTS = """Make each step's description match what its SQL actually executes.
Prefer a complete, read-only SQLite query that can run independently. The first
step must be a complete query. Return a bare clause only if appending it directly
to the end of the previous step's executable SQL creates one valid query without
reordering or replacing any existing clauses. Otherwise, return the complete
query for that step."""

ROLE_INSTRUCTIONS = {
    "generate_sql": """You are the SQLite query generator in EDM-SQL. Produce one correct, efficient
read-only query from the question, examples, database schema, and any BIRD
evidence. Your role is to propose a candidate, not to compare or select queries.

Similar questions and their SQL queries:
{few_shot}

Database schema and example data, if available:
{schema}
{augmented_section}
BIRD evidence for this question:
If supplied, the evidence is part of the question. Follow its formulas, date
formats, abbreviations, value mappings, and units.
{evidence}

Intent profile for this question:
{intent_profile}

Guidance for the identified intent:
{intent_soft_guidance}

Question to answer: {question}

Requirements:
1. Before writing SQL, map any formulas, date rules, and value mappings in the
   evidence to SQL expressions.
2. Check the scope of each formula, the source of date values, whether to count
   rows or entities, and which columns the question asks for.
3. Return only one JSON object that Python's json.loads can parse. Do not add
   explanations or code fences.

Output format:
{{"SQL":"SELECT..."}}""",
    "direct_schema_links": """You are the direct schema linker in EDM-SQL. Map concepts in the question
and its supplied evidence to database tables and columns. Do not write SQL.

Database schema:
{schema}

Question: {question}
{question_evidence_section}
Requirements: Match by meaning, using the exact table and column names in the
schema. Use supplied evidence to resolve ambiguous terms, and include only
matches supported by the question or its evidence. Return an empty links list
when no match is supported.

Return one JSON object only:
{{"links": [{{"table": "...", "column": "..."}}]}}""",
    "value_mentions": """You are the value-mention extractor in EDM-SQL. Extract search phrases
for retrieving stored SQLite values.
Identify literal names, labels, and other values mentioned or mapped in the
question and its evidence. Keep meaningful multiword names together. Do not
return SQL, table names, column names, or inferred values without support.

Question: {question}

BIRD evidence, if provided:
{evidence}

Return only a JSON object with a list of search phrases, or an empty list
when there are none: {{"mentions": ["..."]}}""",
    "adjudicate": """You are the Level-3 candidate adjudicator in EDM-SQL. The program has already
run execution screening and rule checks. Decide whether to retain the current
SQL or replace it with the new candidate. Do not rewrite either query.

Selection stage: {stage}
SSM compares the initial and schema-enhanced queries. RSM compares the query
selected by SSM with the reconstructed query.

Database schema:
{schema}

Schema augmentation hints:
{augmented_schema}

Current SQL:
{current_sql}

New candidate SQL:
{new_sql}

Execution record for the current SQL:
{current_execution}

Execution record for the new SQL:
{new_execution}

Differences recorded by the program:
{difference}

Programmatic risk warnings:
{warnings}

Full memory pool for this stage:
{memory_pool}

Question to answer: {question}
{question_evidence_section}
Requirements:
1. Identify the requested outputs, the unit of counting or aggregation, and
   the scope of any conditions. Account for any requirements in the question evidence.
2. In SSM, check whether the candidate corrects a schema binding or output
   error. In RSM, check whether reconstruction fixes an existing defect without
   altering joins, filters, or aggregates in ways the question does not support.
3. Treat warnings and memory cards as guidance. Focus on three to five
   directly relevant cards if available, or fewer when appropriate. Execution
   success and agreement between sampled results cannot establish correctness
   by themselves.
4. Return accept only if the new query fixes a specific defect in the current
   query without introducing unsupported changes. Otherwise, return retain.
5. In defect, name the specific defect or state that none was established.
   In evidence, explain which requirement and SQL or execution evidence
   support your decision.

Return one JSON object only, without explanations or code fences.
The verdict must be accept or retain:
{{"verdict": "retain", "defect": "...", "evidence": "..."}}""",
    "summarize": """You are the memory summarizer in EDM-SQL. After a candidate decision has been
made, use its comparison record and outcome verdict to propose one reusable
guidance card. Do not reopen the decision or update the memory pool yourself.

Selection stage: {stage}

Current SQL:
{current_sql}

New candidate SQL:
{new_sql}

Selected SQL:
{selected_sql}

Selection action: {selection}

Candidate comparison and execution evidence:
{evidence}

Decision rationale:
{rationale}

Outcome verdict for selected SQL (true = correct, false = incorrect): {outcome_verdict}

Question from the completed decision: {question}
{question_evidence_section}
Requirements:
1. The verdict evaluates only the selected SQL. Do not use it as evidence
   that the unselected SQL was correct or incorrect.
2. Describe when the pattern applies, what risk it poses, and what evidence
   the selector should check next time. Treat question-specific mappings,
   formulas, and literals as local facts, not as rules for other questions.
   Express the lesson without relying on question-specific table or column names.
3. If the evidence does not support a reusable lesson, do not propose a card.

Return one JSON object only:
{{"card": {{"applies_when": "...", "risk": "...", "guidance": "..."}}}}
If no lesson is supported, return {{"card": null}}.""",
    "update": """You are the memory updater in EDM-SQL. Decide how to incorporate the proposed
card into this stage's memory and return the full resulting pool. Do not
reassess the SQL queries that produced the card.

Selection stage: {stage}

Full existing memory pool:
{pool}

Card proposed by the summarizer:
{card}

Maximum number of cards: {capacity}

Requirements: Add distinct guidance, merge overlapping guidance, revise
existing cards when warranted, or remove redundant or misleading cards.
Preserve unrelated guidance. Return the pool unchanged when no update is
warranted, and stay within the stated capacity.

Return one JSON object only. The pool must contain the full resulting memory,
not just the changes:
{{"pool": [{{"applies_when": "...", "risk": "...", "guidance": "..."}}]}}""",
    "decompose": """You are the SQL decomposition module in EDM-SQL. Break the existing query
into an ordered list of executable steps for verification. Describe the query
as it is; do not correct it at this stage.

Requirements: Preserve the query's logic and dependencies. Give each step a
distinct purpose that can be checked, such as selecting the eligible rows or
computing a numerator. Keep operations together when they serve one purpose.
Use a single step only when there is no useful intermediate result to check.
{step_sql_requirements}

Synthetic example (illustrates step granularity only):
Schema: demo_members(member_id PRIMARY KEY), demo_events(member_id).
Question: What percentage of members have at least one event?
SQL to decompose:
SELECT 100.0 * COUNT(DISTINCT e.member_id) / NULLIF(COUNT(DISTINCT m.member_id), 0) AS percentage
FROM demo_members m LEFT JOIN demo_events e ON m.member_id = e.member_id;
Example step list:
{{"steps": [
  {{"description": "Count all members for the denominator.", "sql": "SELECT COUNT(*) FROM demo_members;"}},
  {{"description": "Count members with an event, once per member.", "sql": "SELECT COUNT(DISTINCT e.member_id) FROM demo_members m JOIN demo_events e ON m.member_id = e.member_id;"}},
  {{"description": "Compute the percentage using all members as the denominator.", "sql": "SELECT 100.0 * COUNT(DISTINCT e.member_id) / NULLIF(COUNT(DISTINCT m.member_id), 0) AS percentage FROM demo_members m LEFT JOIN demo_events e ON m.member_id = e.member_id;"}}
]}}

Now decompose the actual input using only its database schema and context.
Database schema:
{schema}

Schema augmentation hints:
{augmented_schema}

SQL to decompose:
{current_sql}

Question: {question}
{question_evidence_section}
Return one JSON object only:
{{"steps": [{{"description": "...", "sql": "..."}}]}}""",
    "check_step": """You are the step verifier in EDM-SQL. Decide whether the indicated step
passes and, if it fails, identify the issue. Use the complete list and earlier
records as context, but judge only the indicated step. Do not edit the list.

Database schema:
{schema}

Schema augmentation hints:
{augmented_schema}

Complete SQL steps list:
{steps}

Step index to verify (zero-based): {index}

Execution records through the indicated step (its record is last):
{execution_records}

Question: {question}
{question_evidence_section}
Requirements: Use the question and its evidence to determine what this step
should do, then check whether the description and SQL implement that operation.
Check the scope of comparisons and exclusions against the intended set of
records. Intermediate results need not have the final answer's shape.
{step_evidence_checks}Set pass to false on execution failure or when you find a specific mismatch.
State what needs correction in diagnostic so the reviser can act on it.

Return one JSON object only. The pass field must be a Boolean:
{{"pass": false, "diagnostic": "..."}}""",
    "revise_steps": """You are the step-list reviser in EDM-SQL. The verifier has identified a
failed step. Use its execution feedback and diagnosis to repair the list.
Return the full revised list, not a final SQL query.

Database schema:
{schema}

Schema augmentation hints:
{augmented_schema}

Complete SQL steps list before revision:
{steps}

Failed step index (zero-based): {index}

Execution record for the failed step:
{execution}

Diagnostic from the verifier:
{diagnostic}

Question: {question}
{question_evidence_section}
Requirements: Correct the offending description or SQL using the question,
evidence, and schema. Change dependent steps only when necessary and preserve
unaffected logic. If another verification round runs, it starts at Step 1 of
the revised list.
{step_sql_requirements}

Return one JSON object only:
{{"steps": [{{"description": "...", "sql": "..."}}]}}""",
    "synthesize": """You are the SQL synthesizer in EDM-SQL. Turn the latest step list into one
complete SQLite query. This query is a candidate for the selector to assess;
you do not decide whether it replaces the current query.

Database schema:
{schema}

Schema augmentation hints:
{augmented_schema}

Latest SQL steps list:
{steps}

Question to answer: {question}
{question_evidence_section}
Requirements: Combine the steps in order into a complete read-only query.
Preserve the supported logic and the requirements in the question and evidence.
Do not introduce joins, filters, or aggregates unsupported by the step list
or the task requirements.

Return one JSON object only:
{{"sql": "SELECT ..."}}""",
    "guarded_repair": """You are the local risk repair module in EDM-SQL. The program has flagged
possible issues in the selected query. Check the flags and propose the smallest
local repair only when there is clear evidence for it.

Database schema:
{schema}

Schema augmentation hints:
{augmented_schema}

SQL to inspect:
{sql}

Risks flagged by the program:
{risks}

Question: {question}
{question_evidence_section}
Requirements: Treat flags as cues, not confirmed defects. Repair only when the
question, evidence, and schema support a specific change. Preserve the requested
semantics and unaffected clauses. Do not rewrite the whole query.

Return only {{"sql": "SELECT ..."}} if a repair is justified, or
{{"sql": null}} otherwise.""",
    "execution_repair": """You are the execution-error repair module in EDM-SQL. Use the actual error
from a failed SQLite query to propose one local repair. Do not generate a new
answer from scratch.

Database schema:
{schema}

Schema augmentation hints:
{augmented_schema}

SQL that failed to execute:
{sql}

Execution error:
{error}

Question: {question}
{question_evidence_section}
Requirements: Make the smallest supported change that addresses the error.
Preserve the requested outputs, conditions, and semantics in the question and
evidence. If you cannot make a reliable local repair, do not substitute a
different query just to get it to run.

Return only {{"sql": "SELECT ..."}} for a supported repair, or
{{"sql": null}} otherwise.""",
}


def messages_for(role, payload):
    def display(value):
        if value is None or value == "":
            return "(not provided)"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    system, _, template = ROLE_INSTRUCTIONS[role].partition("\n\n")
    fields = {name: display(value) for name, value in payload.items()}
    fields.setdefault("augmented_schema", display(None))
    fields["step_sql_requirements"] = STEP_SQL_REQUIREMENTS
    question_evidence = payload.get("question_evidence")
    fields["question_evidence_section"] = (
        f"\nBIRD evidence supplied with the question:\n{display(question_evidence)}\n"
        if question_evidence else ""
    )
    if role == "check_step":
        fields["step_evidence_checks"] = (
            "When the supplied evidence defines a formula, denominator, date format or granularity,\n"
            "value mapping, unit or conversion, or aggregation grain, verify that the current\n"
            "step implements each requirement relevant to its operation.\n"
            if question_evidence else ""
        )
    if role == "generate_sql":
        for name in GENERATION_CONTEXT_FIELDS:
            fields.setdefault(name, display(None))
        augmented = payload.get("augmented_schema")
        fields["augmented_section"] = (
            "\nSchema augmentation hints (verify against the question and schema):\n"
            + display(augmented) + "\n" if augmented else ""
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": template.format(**fields)},
    ]
