from pathlib import Path

REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data"


def to_path(fpath: str | Path) -> Path:
    if not isinstance(fpath, Path):
        return Path(fpath)
    return fpath
