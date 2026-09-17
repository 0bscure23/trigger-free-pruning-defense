#!/usr/bin/env python3
"""Run a Python script after eagerly creating a tiny CUDA context.

This keeps a stage visible in nvidia-smi during Python-side imports and setup
without reserving large amounts of memory.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


HELD_TENSORS = []


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() not in {"", "0", "false", "no", "off"}


def hold_cuda_context() -> None:
    if not _truthy(os.environ.get("EARLY_CUDA_CONTEXT", "1")):
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[cuda_context_runner] torch import failed: {exc}", file=sys.stderr)
        return
    if not torch.cuda.is_available():
        print("[cuda_context_runner] CUDA unavailable; continuing without early context", file=sys.stderr)
        return

    mib = max(int(os.environ.get("CUDA_CONTEXT_MIB", "1")), 1)
    elements = max((mib * 1024 * 1024) // 2, 1)
    count = torch.cuda.device_count()
    for idx in range(count):
        with torch.cuda.device(idx):
            tensor = torch.empty(elements, dtype=torch.float16, device=f"cuda:{idx}")
            tensor[0] = 0
            HELD_TENSORS.append(tensor)
    torch.cuda.synchronize()
    print(
        f"[cuda_context_runner] held {mib} MiB tensor on each of {count} visible CUDA device(s)",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        print("usage: cuda_context_runner.py -- <script.py> [args...]", file=sys.stderr)
        return 2
    sep = argv.index("--")
    target_argv = argv[sep + 1 :]
    if not target_argv:
        print("cuda_context_runner.py: missing target script", file=sys.stderr)
        return 2

    script = Path(target_argv[0])
    sys.argv = target_argv
    hold_cuda_context()
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
