#!/usr/bin/env python3
"""
Robot-end helper: sync the robot chat history file to the local motion computer.

Run this on the robot end when chat_history is produced there, while
realtime_chat_nonverbal_from_txt.py runs on the local end.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# Edit these defaults on the robot if your local machine/user/path is different.
# The robot source file may be .json or .jsonl. The local destination defaults
# to .json because realtime_chat_nonverbal_from_txt.py watches that path first.
DEFAULT_SOURCE = "../chat_history.json"
DEFAULT_DEST = "ywguo@linux:/home/ywguo/Documents/Sophia_VLA/chat_history.json"
DEFAULT_INTERVAL_SEC = 0.5


def resolve_source(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def run_scp(source: Path, dest: str, timeout: float, dry_run: bool, verbose: bool) -> bool:
    if not dest:
        raise ValueError("destination is empty; set DEFAULT_DEST or pass --dest")
    if shutil.which("scp") is None:
        raise RuntimeError("scp command not found; install openssh-client on the robot")

    cmd = ["scp"]
    if verbose:
        cmd.append("-v")
    cmd.extend(["-o", "ConnectTimeout=5", str(source), dest])
    if dry_run:
        print("[DRY]", " ".join(cmd), flush=True)
        return True

    try:
        result = subprocess.run(cmd, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[WARN] scp timed out after {timeout:.1f}s", flush=True)
        return False

    if result.returncode != 0:
        print(
            f"[WARN] scp failed with code {result.returncode}. "
            "Check the scp/ssh error printed above.",
            flush=True,
        )
        return False
    return True


def sync_loop(
    source: Path,
    dest: str,
    interval: float,
    timeout: float,
    once: bool,
    dry_run: bool,
    verbose: bool,
) -> int:
    print(f"[sync] source = {source}", flush=True)
    print(f"[sync] dest   = {dest}", flush=True)
    print(f"[sync] poll   = {interval:.2f}s", flush=True)

    last_signature: tuple[int, int] | None = None
    missing_reported = False

    while True:
        if not source.exists():
            if not missing_reported:
                print(f"[sync] waiting for source file: {source}", flush=True)
                missing_reported = True
            if once:
                return 1
            time.sleep(interval)
            continue

        missing_reported = False
        try:
            signature = file_signature(source)
        except OSError as exc:
            print(f"[WARN] cannot stat source file: {exc}", flush=True)
            if once:
                return 1
            time.sleep(interval)
            continue

        if signature != last_signature:
            ok = run_scp(source, dest, timeout=timeout, dry_run=dry_run, verbose=verbose)
            if ok:
                last_signature = signature
                print(
                    f"[sync] copied {source.name} ({signature[1]} bytes)",
                    flush=True,
                )
            elif once:
                return 1

        if once:
            return 0
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the robot chat history file and scp it to the local motion computer."
    )
    parser.add_argument(
        "--source",
        default=os.getenv("SOPHIA_CHAT_HISTORY_SOURCE", DEFAULT_SOURCE),
        help="Robot-side chat history path, for example ../chat_history.json. Relative paths are resolved next to this script.",
    )
    parser.add_argument(
        "--dest",
        default=os.getenv("SOPHIA_CHAT_HISTORY_REMOTE_DEST", DEFAULT_DEST),
        help="scp destination watched by local realtime_chat_nonverbal_from_txt.py.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("SOPHIA_CHAT_HISTORY_SYNC_INTERVAL", DEFAULT_INTERVAL_SEC)),
        help="Polling interval in seconds.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="scp timeout in seconds.")
    parser.add_argument("--once", action="store_true", help="Copy once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print scp command without copying.")
    parser.add_argument("--verbose", action="store_true", help="Run scp with -v for SSH debugging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = resolve_source(args.source)
    try:
        return sync_loop(
            source=source,
            dest=args.dest,
            interval=max(args.interval, 0.1),
            timeout=args.timeout,
            once=args.once,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n[sync] stopped", flush=True)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
