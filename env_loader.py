import os
from pathlib import Path
from typing import Optional


def load_dotenv(path: Optional[str] = None) -> None:
    p = Path(path or ".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
        
__all__ = ["load_dotenv"]
