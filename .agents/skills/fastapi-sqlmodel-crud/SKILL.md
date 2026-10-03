---
name: fastapi-sqlmodel-crud
description: Add or modify FastAPI routes, SQLModel models, and async SQLite queries in this project following its layered-model, soft-delete, and Jinja2/Alpine.js conventions. Use when editing main.py, API endpoints, database models, or templates under src/.
---

# FastAPI + SQLModel CRUD conventions

This project is a FastAPI app backed by async SQLModel/SQLite (`aiosqlite`) with a
Jinja2 + Alpine.js frontend. Follow these conventions when adding or changing
endpoints, models, and templates.

## Tooling (always use `uv`)

- Install deps: `uv add <package>` (never `pip`). Python is pinned to 3.14 via
  `.python-version` and `requires-python`.
- Dev server: `uv run fastapi dev` (from the repo root).
- Run a script: `uv run python <script>.py`.
- There is no test suite configured; verify by starting the dev server and
  exercising the endpoint (e.g. `curl`).

## Model layering

Define models as separate classes; do not reuse the table model as the request
or response schema:

- `<Name>Base(SQLModel)` — shared fields (`content` is indexed; carry
  `created_at`, `updated_at`, `deleted_at`).
- `<Name>Entity(<Name>Base, table=True)` — adds `id: int | None = Field(default=None, primary_key=True)`.
- `<Name>Public(<Name>Base)` — the response shape (adds `id: int`).
- `<Name>Update(SQLModel)` — partial-update payload.

Timestamps come from the module-level `utcnow()` helper (`datetime.now(UTC)`).

## Endpoint pattern

Inject the DB session with the shared alias, not a raw dependency:

```python
SessionDep = Annotated[AsyncSession, Depends(get_session)]

@app.get("/api/things/{thing_id}")
async def read_thing(thing_id: int, session: SessionDep) -> ThingEntity:
    thing = await session.get(ThingEntity, thing_id)
    if not thing or thing.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Thing not found")
    return thing
```

- Create: `session.add(...)`, `await session.commit()`, `await session.refresh(...)`, return the entity.
- Patch: load, 404 if missing, `obj.sqlmodel_update(payload.model_dump(exclude_unset=True))`,
  `session.add(obj)`, commit, refresh.
- List: build a `select(...)` with `.where(col(Entity.deleted_at).is_(None))`,
  `.offset(offset).limit(limit)`, and cap `limit` with `Annotated[int, Query(le=100)]`.

## Soft delete

Deletes are soft: set `deleted_at = utcnow()` and return, do **not** call `session.delete`.
All read paths must filter out rows where `deleted_at` is not `None`.

## Templates and static assets

- Templates live in `src/templates`, rendered with
  `templates.TemplateResponse(request, "name.html")`.
- Static files are mounted at `/static` from `src/static`; reference them as
  `/static/<file>`.
- Alpine.js is loaded from CDN in the template head and used via inline
  `x-data="{ ... }"` component attributes — keep JS in the template, there is no
  frontend build step.

## Data / scratch scripts

- The character-level GPT experiments live in `experiments/dummy-gpt/` and are not
  part of the notes app; see `experiments/dummy-gpt/README.md` for details.
- `experiments/dummy-gpt/chat-message-body-to-txt.py` and `dedupe-messages.py` are
  standalone scripts with hardcoded relative filenames (`chat.json` -> `messages.txt`
  -> `messages_deduped.txt`). Run them from inside `experiments/dummy-gpt/`; their
  outputs are gitignored.
- `experiments/dummy-gpt/dummy-gpt.py` is a from-scratch autograd/transformer trainer
  (no external ML deps). Treat it as experimental unless the task targets it.
