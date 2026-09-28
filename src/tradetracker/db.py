"""Postgres connection and a minimal SQL migration runner."""

from pathlib import Path

import psycopg

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def connect(url: str) -> psycopg.Connection:
    return psycopg.connect(url)


def migrate(conn: psycopg.Connection, directory: Path = MIGRATIONS) -> list[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    applied = []
    for path in sorted(directory.glob("*.sql")):
        if path.stem in done:
            continue
        sql = path.read_text().replace(
            "CREATE TABLE schema_migrations", "CREATE TABLE IF NOT EXISTS schema_migrations"
        )
        with conn.transaction():
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
        applied.append(path.stem)
    conn.commit()
    return applied
