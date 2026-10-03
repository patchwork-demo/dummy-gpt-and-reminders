# Notes app

A small full-stack notes app: a FastAPI + SQLite backend and an Alpine.js frontend.

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI (async) |
| Models / DB | SQLModel on top of SQLAlchemy async + `aiosqlite` (SQLite file `database.db`) |
| Templating | Jinja2 (`src/templates`) |
| Frontend | Alpine.js (loaded from CDN), `nord.css` + `style.css` |
| Packaging / run | `uv` |

## Features

- Create, list, read, update and delete memos.
- **Soft delete**: `DELETE` only stamps `deleted_at`; deleted memos are hidden from
  the list and from single-item reads, but stay in the database. There is no hard
  delete or restore endpoint.
- `content` is indexed; list results are paginated with `offset` / `limit` (max 100).
- Timestamps are stored in UTC.

## Project layout

```
main.py                     # FastAPI app, models and all routes
src/
  templates/index.html      # Jinja2 page wired with Alpine.js
  static/nord.css           # theme
  static/style.css          # app styles
  dummy_gpt/                # unrelated leftover package
database.db                 # SQLite file (created on first start, gitignored)
experiments/dummy-gpt/     # separate character-level GPT experiments (see its torch.md)
```

## Run

Install dependencies and start the dev server (auto-reload, docs at `/docs`):

```sh
uv run fastapi dev
```

The SQLite database and tables are created automatically on startup.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the notes page (`index.html`) |
| `POST` | `/api/memos/` | Create a memo (`content` required) |
| `GET` | `/api/memos/` | List memos, `?offset=0&limit=100` (excludes deleted) |
| `GET` | `/api/memos/{id}` | Get one memo (`404` if missing or deleted) |
| `PATCH` | `/api/memos/{id}` | Update a memo's `content` |
| `DELETE` | `/api/memos/{id}` | Soft-delete a memo (sets `deleted_at`) |

Models: `MemoBase` (fields) → `MemoEntity` (table) → `MemoPublic` (read) and
`MemoUpdate` (patch payload).

## Frontend status

`src/templates/index.html` uses Alpine.js with the notes kept in an in-memory
`x-data` object — adding and deleting notes works only in the browser and does **not**
call the API or persist anything yet. The backend endpoints above are ready to be
wired up.

## Other

- Character-level GPT experiments live in `experiments/dummy-gpt/` (guide: `torch.md`).
- `uv` docs: https://docs.astral.sh/uv/getting-started/installation/
