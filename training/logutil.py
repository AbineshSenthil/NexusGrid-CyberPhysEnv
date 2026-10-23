"""Mirror stdout/stderr to a UTF-8 log file (Windows PowerShell redirection writes UTF-16)."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional


class _Tee(io.TextIOBase):
    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, data):
        try:
            self._stream.write(data)
        except UnicodeEncodeError:
            self._stream.write(data.encode("ascii", "replace").decode("ascii"))
        self._handle.write(data)
        self._handle.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._handle.flush()


def tee_output(path: Optional[str]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)


def windows_compile_guard() -> None:
    """
    torch.compile needs the MSVC compiler (cl.exe) on Windows. Without it, Unsloth's compiled
    helpers fail at the first training step, so fall back to eager mode. Call before importing torch.
    """
    import os
    import shutil

    if os.name == "nt" and shutil.which("cl") is None and "TORCHDYNAMO_DISABLE" not in os.environ:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        print("[nexusgrid] cl.exe not found: torch.compile disabled (eager mode)", flush=True)


def quiet_transformers() -> None:
    """
    Show only transformers errors. Qwen3 ships ``max_length=40960`` in its generation config, so
    every generate() call with ``max_new_tokens`` logs a harmless notice that floods the console.
    """
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass
