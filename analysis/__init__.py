"""Analysis package — data logging and paper-analysis tooling.

Per docs/data-pipeline.md:
- Terminal (bot-vs-bot) data logs to JSONL files under data/terminal/
- Web (human-vs-bot) data logs to PostgreSQL (see db/schema.sql)
- Both sources share the unified record format for paper analysis
"""
