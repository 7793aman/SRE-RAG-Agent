"""`scripts/seed_db.py` — the seed orchestrator.

The `--no-ingest` path (migrations + demo users) end to end against live
Postgres; skips (via `clean_users` → `db_ready`) if Postgres is unreachable.
"""

from __future__ import annotations

import pytest

from app import db
from app.middleware.auth import verify_password
from scripts import seed_db


def _users() -> dict[str, tuple[str, bool]]:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT username, password_hash, is_admin FROM users")
        return {r.username: (r.password_hash, r.is_admin) for r in cur.fetchall()}


def test_seed_users_creates_every_demo_user(clean_users: None) -> None:
    seed_db.seed_users()
    users = _users()
    assert set(users) == {username for username, _, _ in seed_db.DEMO_USERS}
    for username, _, is_admin in seed_db.DEMO_USERS:
        assert users[username][1] is is_admin


def test_seed_users_passwords_verify(clean_users: None) -> None:
    seed_db.seed_users()
    users = _users()
    assert verify_password("agent123", users["agent@demo.local"][0])
    assert verify_password("admin123", users["admin@demo.local"][0])


def test_seed_users_is_idempotent(clean_users: None) -> None:
    seed_db.seed_users()
    seed_db.seed_users()
    assert len(_users()) == len(seed_db.DEMO_USERS)


def test_seed_users_refreshes_an_existing_row(clean_users: None) -> None:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (%s, %s, %s)",
            ("admin@demo.local", "stale-hash", False),
        )
    seed_db.seed_users()
    hash_, is_admin = _users()["admin@demo.local"]
    assert is_admin is True
    assert verify_password("admin123", hash_)


def test_main_no_ingest_returns_zero_and_seeds(clean_users: None) -> None:
    assert seed_db.main(["--no-ingest"]) == 0
    assert set(_users()) == {username for username, _, _ in seed_db.DEMO_USERS}


@pytest.mark.parametrize("bad", ["lots", "-1", "3.5"])
def test_noise_sample_arg_rejects_bad_values(bad: str) -> None:
    with pytest.raises(SystemExit):
        seed_db.main(["--no-ingest", "--noise-sample", bad])


@pytest.mark.parametrize(("raw", "expected"), [("all", "all"), ("300", 300), ("0", 0)])
def test_noise_sample_arg_accepts_valid_values(raw: str, expected: object) -> None:
    assert seed_db._parse_noise_sample(raw) == expected
