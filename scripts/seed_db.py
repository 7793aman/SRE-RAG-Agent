"""Seed orchestrator: schema migrations, the two demo users, and (optionally) the
document corpus ingestion.

    uv run python scripts/seed_db.py                     # everything
    uv run python scripts/seed_db.py --no-ingest         # DB only, no vector store
    uv run python scripts/seed_db.py --noise-sample 500  # 47 signal + 500 noise
    uv run python scripts/seed_db.py --noise-sample all  # 47 signal + all noise

The signal corpus (`seed/docs/true_data/`, 47 files) is always ingested in full.
The noise corpus (`seed/docs/noisy_data/`) is sampled to a configurable size with
a fixed seed, so the same `--noise-sample N` always picks the same N files.

Corpus wiring: the seeder reads noise from `seed/docs/noisy_data/`. If that's
empty, `stage_noise_corpus()` symlinks files in from a root-level staging
folder (`noisy_data 2/`, or `noisy_data/`) — see `seed/docs/README.md`.

Doc ingestion depends on the embedding + vector-store services (ticket #22); until
those land, run with `--no-ingest`.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from app.db import connection, run_migrations
from app.middleware.auth import hash_password

_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = _ROOT / "seed" / "docs"

SIGNAL_SUBDIR = "true_data"
NOISE_SUBDIR = "noisy_data"

# Parsers the document processor (ticket #22) can handle.
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}

# Fixed seed for noise sampling — stable selection across runs and machines.
NOISE_SAMPLE_SEED = 42

# docling's PDF pipeline runs a full layout-analysis model over every page, on
# CPU. A handful of multi-hundred-page manuals in the noise corpus (one is
# 1,324 pages) can each take longer to ingest than the entire rest of a sample
# combined. Excluding anything over this budget keeps a seed run's duration
# roughly proportional to the file count you asked for, instead of being at
# the mercy of which few outliers a random sample happened to draw. `None`
# disables the filter entirely (e.g. for a deliberate full-corpus run).
DEFAULT_MAX_NOISE_PAGES: int | None = 100

# Root-level staging folders to pull the noise corpus from, in preference order,
# when `seed/docs/noisy_data/` is empty. `noisy_data 2/` is what a fresh drop of
# the corpus tends to land as (a second copy alongside a stale `noisy_data/`).
_NOISE_STAGING_CANDIDATES = ("noisy_data 2", "noisy_data")

# username, plaintext password, is_admin
DEMO_USERS: tuple[tuple[str, str, bool], ...] = (
    ("agent@demo.local", "agent123", False),
    ("admin@demo.local", "admin123", True),
    ("7793aman", "7793@aman", True),
)


@dataclass(frozen=True)
class CorpusSelection:
    """The files a seed run will ingest: all signal, a sample of noise."""

    signal: list[Path]
    noise: list[Path]

    @property
    def total(self) -> int:
        return len(self.signal) + len(self.noise)


def _list_docs(directory: Path) -> list[Path]:
    """Every supported document under ``directory``, sorted for a stable order."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and p.name != ".gitkeep"
    )


def stage_noise_corpus(root: Path = _ROOT) -> int:
    """Wire a root-level noise staging folder into `seed/docs/noisy_data/`.

    A no-op when `seed/docs/noisy_data/` already has files. Otherwise, for the
    first existing candidate in `_NOISE_STAGING_CANDIDATES`, symlinks each of
    its files in — a symlink, not a copy, so the ~800MB corpus body is never
    duplicated on disk. Returns the number of files newly linked.
    """
    target = root / "seed" / "docs" / NOISE_SUBDIR
    if _list_docs(target):
        return 0
    target.mkdir(parents=True, exist_ok=True)
    for name in _NOISE_STAGING_CANDIDATES:
        source = root / name
        if not source.is_dir() or source.resolve() == target.resolve():
            continue
        files = _list_docs(source)
        if not files:
            continue
        for f in files:
            (target / f.name).symlink_to(f.resolve())
        logger.info("staged {} noise files from {} -> {}", len(files), source, target)
        return len(files)
    return 0


def _pdf_page_count(path: Path) -> int | None:
    """Page count via the page tree only (no content parsing) — cheap.

    Returns ``None`` for a non-PDF, or a PDF whose page count can't be read
    (e.g. corrupt) — callers should treat that as "unknown," not "too long."
    """
    if path.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001 — an unreadable PDF fails open, not closed
        return None


def _within_page_budget(path: Path, max_pages: int | None) -> bool:
    if max_pages is None:
        return True
    pages = _pdf_page_count(path)
    return pages is None or pages <= max_pages


def select_corpus(
    noise_sample: int | Literal["all"],
    docs_dir: Path = DOCS_DIR,
    max_noise_pages: int | None = DEFAULT_MAX_NOISE_PAGES,
) -> CorpusSelection:
    """Pick the corpus for a seed run.

    Signal is always taken in full. The noise pool first drops any PDF over
    ``max_noise_pages`` (``None`` disables this), then is sampled to
    ``noise_sample`` files with a fixed seed; ``"all"`` (or a count at/above
    the pool size) takes every eligible noise file, ``0`` takes none.
    """
    signal = _list_docs(docs_dir / SIGNAL_SUBDIR)
    noise_pool = [
        p for p in _list_docs(docs_dir / NOISE_SUBDIR) if _within_page_budget(p, max_noise_pages)
    ]

    if noise_sample == "all":
        noise = noise_pool
    else:
        n = int(noise_sample)
        if n < 0:
            raise ValueError(f"noise_sample must be >= 0 or 'all', got {noise_sample!r}")
        if n >= len(noise_pool):
            noise = noise_pool
        else:
            noise = sorted(random.Random(NOISE_SAMPLE_SEED).sample(noise_pool, n))

    return CorpusSelection(signal=signal, noise=noise)


def seed_users(users: Iterable[tuple[str, str, bool]] = DEMO_USERS) -> None:
    """Upsert the demo users. Idempotent — re-running refreshes their hashes."""
    with connection() as conn, conn.cursor() as cur:
        for username, password, is_admin in users:
            cur.execute(
                """
                INSERT INTO users (username, password_hash, is_admin)
                VALUES (%s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET
                    password_hash = EXCLUDED.password_hash,
                    is_admin = EXCLUDED.is_admin
                """,
                (username, hash_password(password), is_admin),
            )
            logger.info("seeded user {} (admin={})", username, is_admin)


def ingest_corpus(selection: CorpusSelection) -> None:
    """Parse, embed, and upsert every selected file into the vector store.

    Imports the ingestion services lazily so `--no-ingest` runs (and this
    module's import) don't need ticket #22's dependencies.
    """
    from app.models import RetrievedChunk
    from app.services.document_processor import DocumentProcessor
    from app.services.embedding_service import embed_texts
    from app.services.vector_store import source_exists, upsert_chunks

    processor = DocumentProcessor()
    ordered = [(p, "signal") for p in selection.signal]
    ordered += [(p, "noise") for p in selection.noise]
    logger.info(
        "ingesting {} files ({} signal + {} noise)",
        selection.total,
        len(selection.signal),
        len(selection.noise),
    )
    ingested = skipped = failed = chunk_count = 0
    for idx, (path, label) in enumerate(ordered, start=1):
        try:
            if source_exists(path.name):
                logger.info(
                    "[{}/{}] skip {} {} (already ingested)", idx, selection.total, label, path.name
                )
                skipped += 1
                continue
            meta = processor.process_document(str(path))
            if not meta:
                logger.warning("[{}/{}] {} {} → 0 chunks", idx, selection.total, label, path.name)
                failed += 1
                continue
            chunks = [
                RetrievedChunk(text=m["text"], source=m["source"], page_number=m.get("page_number"))
                for m in meta
            ]
            upsert_chunks(chunks, embed_texts([c.text for c in chunks]))
            ingested += 1
            chunk_count += len(chunks)
        except Exception:  # noqa: BLE001 — one bad file must not abort the seed
            logger.exception("[{}/{}] failed {} {}", idx, selection.total, label, path.name)
            failed += 1
    logger.info(
        "ingestion done — {} files, {} chunks, {} skipped, {} failed",
        ingested,
        chunk_count,
        skipped,
        failed,
    )


def _parse_noise_sample(raw: str) -> int | Literal["all"]:
    if raw == "all":
        return "all"
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"--noise-sample must be a non-negative integer or 'all', got {raw!r}"
        ) from None
    if value < 0:
        raise SystemExit(f"--noise-sample must be a non-negative integer or 'all', got {raw!r}")
    return value


def _parse_max_noise_pages(raw: str) -> int | None:
    if raw == "none":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"--max-noise-pages must be a positive integer or 'none', got {raw!r}"
        ) from None
    if value <= 0:
        raise SystemExit(f"--max-noise-pages must be a positive integer or 'none', got {raw!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the operational DB and document corpus")
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="run migrations + demo users only; skip vector-store ingestion",
    )
    parser.add_argument(
        "--noise-sample",
        default="150",
        metavar="N|all",
        help="how many noise files to ingest (fixed seed); default 150",
    )
    parser.add_argument(
        "--max-noise-pages",
        default=str(DEFAULT_MAX_NOISE_PAGES),
        metavar="N|none",
        help=(
            "exclude noise PDFs over this many pages before sampling, so a few "
            f"multi-hundred-page outliers can't dominate ingestion time; default "
            f"{DEFAULT_MAX_NOISE_PAGES}, 'none' disables the filter"
        ),
    )
    args = parser.parse_args(argv)
    noise_sample = _parse_noise_sample(args.noise_sample)
    max_noise_pages = _parse_max_noise_pages(args.max_noise_pages)

    logger.info("running migrations")
    applied = run_migrations()
    logger.info("applied {} migration(s): {}", len(applied), ", ".join(applied))

    logger.info("seeding demo users")
    seed_users()

    if args.no_ingest:
        logger.info("--no-ingest set; skipping document ingestion")
        return 0

    stage_noise_corpus()
    selection = select_corpus(noise_sample, max_noise_pages=max_noise_pages)
    if selection.total == 0:
        logger.warning("no documents found under {} — nothing to ingest", DOCS_DIR)
        return 0
    ingest_corpus(selection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
