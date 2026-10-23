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
