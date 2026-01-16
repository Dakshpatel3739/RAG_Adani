import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
