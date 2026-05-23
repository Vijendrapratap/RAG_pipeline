"""Re-attempt files that landed in dead_letter/<reason>/.

Per PRD §6 Phase 4: 'resets their status to pending and invokes the main
ingester.' Workflow:

    1. Walk dead_letter/<reason>/*.chunks.json (skip 'oversize' by default
       since those won't fit any better on retry — pass --include-oversize
       to retry them too).
    2. For each file: clear the entry from ingest_status (so the main
       ingester treats it as new) and copy it back to --chunks-dir.
    3. Invoke ingestion.bulk_ingest_hardened.main with --retry-failed.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from ingestion.bulk_ingest_hardened import (
    DEAD_LETTER_DIR,
    PROGRESS_DB_PATH,
    main as bulk_main,
)
from ingestion.utils.progress_db import ProgressDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("retry_dead_letter")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Re-attempt dead-lettered files.")
    p.add_argument("--chunks-dir", required=True,
                   help="Directory the main ingester scans for *.chunks.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--include-oversize", action="store_true",
                   help="Also retry files quarantined as 'oversize'.")
    args = p.parse_args(argv)

    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if not DEAD_LETTER_DIR.is_dir():
        log.info("no dead_letter directory at %s — nothing to retry", DEAD_LETTER_DIR)
        return 0

    db = ProgressDB(PROGRESS_DB_PATH)
    moved = 0
    for reason_dir in sorted(DEAD_LETTER_DIR.iterdir()):
        if not reason_dir.is_dir():
            continue
        if reason_dir.name == "oversize" and not args.include_oversize:
            log.info("skipping %s (use --include-oversize to retry)", reason_dir)
            continue
        for src in sorted(reason_dir.glob("*.chunks.json")):
            db.reset_to_pending(src.name)
            dst = chunks_dir / src.name
            shutil.copy2(src, dst)
            moved += 1
            log.info("requeued %s (from %s)", src.name, reason_dir.name)
    db.close()

    if moved == 0:
        log.info("nothing to retry")
        return 0

    log.info("requeued %d files — invoking bulk ingester with --retry-failed", moved)
    return bulk_main([
        "--chunks-dir", str(chunks_dir),
        "--batch-size", str(args.batch_size),
        "--retry-failed",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
