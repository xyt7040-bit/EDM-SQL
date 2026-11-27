# EDM-SQL supplementary materials

This repository contains the EDM-SQL reference implementation and prompt
templates. To check the overall execution accuracy (EX) on Spider-Dev and
BIRD-Dev, [re-score the saved predictions](#re-score-the-saved-predictions)
without an LLM. See the [result inventory](results/README.md) for file checksums
and coverage.

## Datasets

Download the datasets from their original sources. This repository does not
redistribute the datasets or their SQLite databases.

| Dataset | Download source | Split or release used in the paper |
| --- | --- | --- |
| Spider | [Spider 1.0](https://yale-lily.github.io/spider) | Development and test |
| BIRD | [BIRD-SQL](https://bird-bench.github.io/) | `dev_20240627` (1,534 questions) |
| Spider-DK | [Dataset repository](https://github.com/ygan/Spider-DK) | Robustness benchmark |
| Spider-Syn | [Dataset repository](https://github.com/ygan/Spider-Syn) | Robustness benchmark |
| Spider-Realistic | [Zenodo release](https://zenodo.org/records/5205322) | Robustness benchmark |

## Re-score the saved predictions

Use Python 3.9 or later. Install the evaluator dependencies:

```sh
python -m pip install -r requirements-eval.txt
python -m nltk.downloader punkt_tab
```

Download the Spider and BIRD development data from the sources listed above.

| Data directory | Expected contents |
| --- | --- |
| Spider development data | `dev_gold.sql`, `tables.json`, and `database/` from the dataset |
| BIRD development data (2024-06-27 release) | `dev.sql` (or `dev_gold.sql`), `dev.json`, and `dev_databases/` |

Use the evaluation databases corresponding to the runs in the paper. Run from
the artifact root:

```sh
python evaluate.py --dataset spider-dev --data-dir /path/to/spider
python evaluate.py --dataset bird-dev --data-dir /path/to/bird/dev_20240627
```

Verified overall EX: **89.7%** on Spider-Dev (1,034 questions) and **72.62%**
on BIRD-Dev (1,534 questions). The wrapper opens databases in read-only mode
and checks the input format: one nonblank prediction per Spider question, and
ordered prediction IDs and matching database IDs for BIRD.

## Reference code and prompt templates

Verify the reported Spider-Dev and BIRD-Dev scores with the
[saved predictions and bundled evaluators](#re-score-the-saved-predictions).
Re-scoring requires no model API.

For new inference runs, we recommend Python 3.12. Install the method
dependencies and configure an OpenAI-compatible endpoint via `EDM_SQL_BASE_URL`,
`EDM_SQL_API_KEY`, and `EDM_SQL_MODEL`. The original
`Qwen3-Coder-Plus-2025-09-23` endpoint is no longer available.

### Prepare benchmark inputs

Convert downloaded data to the JSONL consumed by the index builder and runner:

```sh
python prepare_inputs.py --dataset spider-dev --data-dir /path/to/spider --output runs/spider-dev/questions.jsonl
python prepare_inputs.py --dataset bird-dev --data-dir /path/to/bird/dev_20240627 --output runs/bird-dev/questions.jsonl
```

Use `spider-test` for `test.json` and `test_database/` under the Spider root.
The converter preserves source order, uses the zero-based row index when a
question ID is absent, and keeps BIRD evidence as a separate field. Reference
SQL is left out of inference inputs. Prepare other SQLite datasets in the same
JSONL format described below.

### Build the index and run inference

```sh
python -m pip install -r requirements-method.txt
python -X utf8 build_value_index.py runs/spider-dev/questions.jsonl --value-index-dir .value_index/spider-dev
python -X utf8 run_method.py runs/spider-dev/questions.jsonl --value-index-dir .value_index/spider-dev
```

For value-based schema linking, the runner defaults to the local
`Qwen/Qwen3-Embedding-0.6B` model. Pass `--embedding-model` (or set
`EDM_SQL_EMBEDDING_MODEL`) to select a model ID or point to a downloaded
model directory. Compatible replacement models must define the `query`
prompt in their Sentence Transformers configuration. Use
`--embedding-device` or `EDM_SQL_EMBEDDING_DEVICE` to choose `cpu` or a
CUDA device explicitly. For example:

```sh
python -X utf8 build_value_index.py questions.jsonl --embedding-model /path/to/Qwen3-Embedding-0.6B --embedding-device cpu
python -X utf8 run_method.py questions.jsonl --embedding-model /path/to/Qwen3-Embedding-0.6B --embedding-device cpu
```

Build the value index locally with `build_value_index.py` before running
inference. This step embeds distinct SQLite text values by column and does not
require the generation API. Both commands use `.value_index` by default. If
you set `--value-index-dir` or `--embedding-model`, use the same settings for
both commands. Rebuild the index after moving or changing a database or
switching embedding models. Use separate directories for dataset splits
that share database IDs.

During inference the generation model extracts search phrases from the question
and any BIRD evidence. Only those phrases are embedded; their vectors are
compared with the saved database-value vectors using cosine similarity, returning
up to five values per column. If the extractor response is missing or malformed,
the runner splits the question and evidence into search terms and uses the same
vector index. An explicit empty list skips value retrieval. Supplying `augmented_schema` or
`schema_sql` bypasses this retrieval for that question.

Each nonblank JSONL line requires `question` and `database` (SQLite path,
absolute or relative to the input file), with an optional `id`. For
SQL generation, the input may also include `few_shot`, `evidence`,
`intent_profile`, and `intent_soft_guidance`. The template renders these
fields as separate sections; missing values are marked as not provided.
To use few-shot examples, add `few_shot` to each relevant record after running
`prepare_inputs.py`, as a list of `{"question": "...", "sql": "SELECT ..."}` objects.
The generation template requests a JSON response with the `SQL` key. Question
evidence is also passed to direct schema linking, semantic adjudication,
reconstruction, final repair, and memory summarization, separately from any
SQL execution evidence.
For pre-generated candidates, pass `initial_sql` and/or `schema_sql`; if available,
pass their three signals in the `augmented_schema` object. Output is JSONL,
with the two stage decisions, delivery status and result. See
[the reference code](edm_sql/pipeline.py) and [prompt templates](edm_sql/prompts.py).

### Feedback and memory updates

In the paper's benchmark evaluation, an execution-based evaluator provides
binary correctness labels. For a completed question, a label on a Level-3
decision can update that stage's memory for subsequent questions.

The reference CLI accepts feedback through interactive human input. Memory is
disabled by default. Add `--memory` to enable feedback with separate, initially
empty SSM and RSM pools.

```sh
python -X utf8 run_method.py questions.jsonl --memory
python -X utf8 run_selector.py pairs.jsonl --memory
```

For each displayed Level-3 selection, enter `1` (correct), `0` (incorrect), or
`s` (skip). The CLI waits before proceeding. Prompts go to stderr; results
remain JSONL on stdout.

### Level-2 programmatic risk checks

Level 2 uses the SQL tree difference and executed output metadata. A hard
rejection requires an explicit question constraint and a new, verifiable
violation; it retains the current SQL. The exit record contains the rule ID,
the matched requirement, and the affected SQL feature and scope.

| Scope | Rule | Rejection condition |
| --- | --- | --- |
| SSM / RSM | `explicit_single_output` | An explicit one-column requirement; the current result has one column and the candidate has multiple columns. |
| SSM / RSM | `changed_row_count` | An explicit request to count all rows including duplicates; replacing `COUNT(*)` with `COUNT(DISTINCT ...)` is the only query change and lowers the executed scalar count. |
| RSM | `added_filter` | The question forbids filtering; a new outer `WHERE` is the only change and removes rows. |
| RSM | `added_aggregation` | The candidate introduces outer grouping or a non-window aggregate that the question explicitly forbids. |
| RSM | `added_deduplication` | The question requires keeping duplicate rows; a new outer `DISTINCT` is the only change and removes rows. |
| RSM | `transformed_output` | The question asks for values as stored; wrapping the same column in `CAST`, `LOWER`, or `UPPER` changes its returned values. Output positions must be comparable and row counts must match; column labels and row order do not count as value changes. |

Changes that do not meet a hard-rejection condition may be passed to Level 3
as scope-aware warnings. The optimizer's value calibration handles uniquely
bound, single-table text equality predicates.

## Run the selector on supplied pairs

```sh
python -X utf8 run_selector.py pairs.jsonl
```

Each JSONL object contains `id` (optional), `stage` (`SSM` or `RSM`),
`question`, `database` (SQLite path relative to the input file or absolute),
`current_sql`, and `new_sql`. Optional fields are `schema`, `augmented_schema`,
and question `evidence`. Output is one JSON object per pair, including the
selected SQL, verdict, exit level, reason, and any collected evidence or rationale.
The selector does not run candidate generation, reconstruction, or embedding
retrieval. The same API variables as `run_method.py` are needed only when a pair
reaches Level 3. This command starts at Level 1; the paper's fixed-pair study
compares decisions on pairs that have already reached Level 3.

## Tests

Run `python -m unittest discover -s tests`.

See [Third-party notices](notices/README.md) for evaluator sources and adaptations.
