"""hermesnote DB 存取（限定可用，含 DDL，見 CLAUDE.md）。量化回測任務/結果存這裡。"""

import os

import asyncpg
from dotenv import load_dotenv

load_dotenv()

HERMESNOTE_DATABASE_URL = os.getenv("HERMESNOTE_DATABASE_URL")


async def get_conn() -> asyncpg.Connection:
    return await asyncpg.connect(HERMESNOTE_DATABASE_URL)
