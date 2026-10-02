from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession


class MemoEntity(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content: str = Field(index=True)


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
def homepage(request: Request) -> Response:
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/memos/")
async def create_memo(memo: MemoEntity, session: SessionDep) -> MemoEntity:
    session.add(memo)
    await session.commit()
    await session.refresh(memo)
    return memo


@app.get("/api/memos/")
async def read_memos(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[MemoEntity]:
    result = await session.exec(select(MemoEntity).offset(offset).limit(limit))
    return list(result.all())


@app.get("/api/memos/{memo_id}")
async def read_memo(memo_id: int, session: SessionDep) -> MemoEntity:
    memo = await session.get(MemoEntity, memo_id)
    if not memo:
        raise HTTPException(status_code=404, detail="Memo not found")
    return memo


@app.delete("/api/memos/{memo_id}")
async def delete_memo(memo_id: int, session: SessionDep):
    memo = await session.get(MemoEntity, memo_id)
    if not memo:
        raise HTTPException(status_code=404, detail="Memo not found")
    await session.delete(memo)
    await session.commit()
    return {"ok": True}
