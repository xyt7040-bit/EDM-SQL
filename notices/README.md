# Third-party notices

- `evaluators/spider/` contains the local experimental copy of the
  [Spider test-suite evaluator](https://github.com/taoyds/test-suite-sql-eval)
  (Apache-2.0). The evaluator may refer to the suite metric; the documented
  command uses the database directory from the corresponding Spider run.
  This package changes SQLite connections to read-only.
- `evaluators/bird/` contains the local experimental copy of the
  [BIRD execution evaluator](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py)
  (MIT). This package adds explicit file paths, UTF-8 input handling, read-only
  SQLite connections, and a serial execution option. Result comparison uses
  the same routine as the archived evaluator.
