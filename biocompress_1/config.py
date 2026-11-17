"""Project defaults. Importing configuration does not read genome files."""

from pathlib import Path

DNA_FILE = "E_Coli"
DNA_FILE_PATH = str(Path(__file__).resolve().parent / "data") + "/"
DNA_FILE_TXT = DNA_FILE + ".txt"
DNA_FILE_FA = DNA_FILE + ".fa"
COMPLEMENT_TABLE = str.maketrans("ACTG", "TGAC")
HEIGHT = 11
COMPARE_LENGTH = 50


def load_content() -> str:
    """Load the default normalized genome only when explicitly requested."""
    return (Path(DNA_FILE_PATH) / DNA_FILE_TXT).read_text(encoding="utf-8")


def __getattr__(name: str) -> str:
    """Preserve legacy `from config import CONTENT` usage without eager I/O."""
    if name == "CONTENT":
        return load_content()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
