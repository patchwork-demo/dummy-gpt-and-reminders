# AGENTS.md

## Environment
- Python 3.14 via `uv`. Always run Python through uv: `uv run ...`
  (`uv run fastapi dev`, `uv run python script.py`). Never invoke bare `python`.
- Shell varies by machine: POSIX `sh` on macOS/Linux, PowerShell 7 on Windows.
  Detect from the session OS; do not assume `sh`.
- Prefer cross-platform commands. Avoid shell-specific escaping and heredocs;
  for anything non-trivial, write a `.py`/`.sh` file instead of a
  `python -c "..."` one-liner.

## Commands
- Dev server (auto-reload, docs at `/docs`): `uv run fastapi dev`
- Add / sync deps: `uv add <pkg>`, `uv sync`
- SQLite file `database.db` is gitignored and created on first startup.

## Conventions
- Backend uses layered SQLModel models: `MemoBase` -> `MemoEntity` (table
  `memoentity`) -> `MemoPublic` / `MemoUpdate`. See the `fastapi-sqlmodel-crud`
  skill before editing routes or models.
- Soft delete only: `DELETE` stamps `deleted_at`. No hard delete or restore.
- Timestamps are stored in UTC.

## Layout
- `main.py` - FastAPI app, models, and all routes
- `src/templates`, `src/static` - Jinja2 + Alpine.js UI
- `experiments/dummy-gpt/` - separate GPT experiments (own `README.md`)
