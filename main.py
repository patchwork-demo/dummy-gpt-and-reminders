from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, SQLModel, col, select
from sqlmodel.ext.asyncio.session import AsyncSession


def utcnow() -> datetime:
    return datetime.now(UTC)


class MemoBase(SQLModel):
    content: str = Field(index=True)
    is_completed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(
        default_factory=utcnow, sa_column_kwargs={"onupdate": utcnow}
    )


class MemoEntity(MemoBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    deleted_at: datetime | None = Field(default=None)


class MemoPublic(MemoBase):
    id: int | None = None


class MemoCreate(SQLModel):
    content: str


class MemoUpdate(SQLModel):
    content: str = ""
    is_completed: bool


sqlite_file_name = "database.db"
sqlite_url = f"sqlite+aiosqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_async_engine(sqlite_url, connect_args=connect_args)


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session():
    async with AsyncSession(engine) as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")


@app.get("/")
async def homepage(request: Request, session: SessionDep) -> Response:
    result = await session.exec(
        select(MemoEntity).where(col(MemoEntity.deleted_at).is_(None))
    )
    memos = [MemoPublic.model_validate(m).model_dump(mode="json") for m in result.all()]
    # print(f'memos: {memos}')
    return templates.TemplateResponse(request, "index.html", {"memos": memos})


@app.post("/api/memos/")
async def create_memo(memo: MemoCreate, session: SessionDep) -> MemoEntity:
    memo_db = MemoEntity(**memo.model_dump())
    session.add(memo_db)
    await session.commit()
    await session.refresh(memo_db)
    return memo_db


@app.patch("/api/memos/{memo_id}")
async def update_memo(
    memo_id: int, memo: MemoUpdate, session: SessionDep
) -> MemoEntity:
    memo_db = await session.get(MemoEntity, memo_id)
    if not memo_db:
        raise HTTPException(status_code=404, detail="Memo not found")
    memo_data = memo.model_dump(exclude_unset=True)
    memo_db.sqlmodel_update(memo_data)
    session.add(memo_db)
    await session.commit()
    await session.refresh(memo_db)
    return memo_db


@app.get("/api/memos/")
async def read_memos(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[MemoEntity]:
    statement = (
        select(MemoEntity)
        .where(col(MemoEntity.deleted_at).is_(None))
        .offset(offset)
        .limit(limit)
    )
    result = await session.exec(statement)
    return list(result.all())


@app.get("/api/memos/{memo_id}")
async def read_memo(memo_id: int, session: SessionDep) -> MemoEntity:
    memo = await session.get(MemoEntity, memo_id)
    if not memo or memo.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Memo not found")
    return memo


@app.delete("/api/memos/{memo_id}")
async def delete_memo(memo_id: int, session: SessionDep):
    memo = await session.get(MemoEntity, memo_id)
    if not memo or memo.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Memo not found")
    memo.deleted_at = utcnow()
    await session.commit()
    return {"ok": True}
