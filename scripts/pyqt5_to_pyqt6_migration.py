import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root


# -----------------------------
# Replacement rules
# -----------------------------

IMPORT_REPLACEMENTS = {
    r"from PyQt5\.QtWidgets import": "from PyQt6.QtWidgets import",
    r"from PyQt5\.QtCore import": "from PyQt6.QtCore import",
    r"from PyQt5\.QtGui import": "from PyQt6.QtGui import",
    r"import PyQt5\.QtWidgets as": "import PyQt6.QtWidgets as",
    r"import PyQt5\.QtCore as": "import PyQt6.QtCore as",
    r"import PyQt5\.QtGui as": "import PyQt6.QtGui as",
}

ENUM_REPLACEMENTS = {
    # Alignment
    r"Qt\.AlignCenter": "Qt.AlignmentFlag.AlignCenter",
    r"Qt\.AlignLeft": "Qt.AlignmentFlag.AlignLeft",
    r"Qt\.AlignRight": "Qt.AlignmentFlag.AlignRight",
    r"Qt\.AlignTop": "Qt.AlignmentFlag.AlignTop",
    r"Qt\.AlignBottom": "Qt.AlignmentFlag.AlignBottom",

    # Orientation
    r"Qt\.Horizontal": "Qt.Orientation.Horizontal",
    r"Qt\.Vertical": "Qt.Orientation.Vertical",

    # Mouse buttons
    r"Qt\.LeftButton": "Qt.MouseButton.LeftButton",
    r"Qt\.RightButton": "Qt.MouseButton.RightButton",
    r"Qt\.MiddleButton": "Qt.MouseButton.MiddleButton",

    # Focus
    r"Qt\.StrongFocus": "Qt.FocusPolicy.StrongFocus",
    r"Qt\.ClickFocus": "Qt.FocusPolicy.ClickFocus",
    r"Qt\.NoFocus": "Qt.FocusPolicy.NoFocus",

    # Item selection
    r"QAbstractItemView\.SingleSelection":
        "QAbstractItemView.SelectionMode.SingleSelection",
    r"QAbstractItemView\.SelectItems":
        "QAbstractItemView.SelectionBehavior.SelectItems",
}

VALIDATOR_REPLACEMENTS = {
    r"QRegExpValidator": "QRegularExpressionValidator",
}

MPL_REPLACEMENTS = {
    r"matplotlib\.use\([\"']Qt5Agg[\"']\)": "matplotlib.use('QtAgg')",
    r"matplotlib\.use\([\"']Qt5Agg[\"'], force=True\)":
        "matplotlib.use('QtAgg', force=True)",
}


# -----------------------------
# Apply replacements to a file
# -----------------------------

def apply_replacements(text: str, replacements: dict) -> str:
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def process_file(path: Path):
    original = path.read_text(encoding="utf-8")

    updated = original
    updated = apply_replacements(updated, IMPORT_REPLACEMENTS)
    updated = apply_replacements(updated, ENUM_REPLACEMENTS)
    updated = apply_replacements(updated, VALIDATOR_REPLACEMENTS)
    updated = apply_replacements(updated, MPL_REPLACEMENTS)

    if updated != original:
        backup = path.with_suffix(path.suffix + ".bak_pyqt5")
        shutil.copy2(path, backup)
        path.write_text(updated, encoding="utf-8")
        print(f"Updated: {path}")
    else:
        print(f"No changes: {path}")


# -----------------------------
# Walk project and update files
# -----------------------------

def main():
    print("Starting PyQt5 → PyQt6 migration...")

    SOURCE_DIRS = [
        ROOT / "calibration",
        ROOT / "analysis",
        ROOT / "tuner",
        ROOT / "scripts",
        ROOT / "tests",
    ]

    for src in SOURCE_DIRS:
        if not src.exists():
            continue
        for pyfile in src.rglob("*.py"):
            if "pyqt5_to_pyqt6_migration" in str(pyfile):
                continue
            try:
                process_file(pyfile)
            except UnicodeDecodeError:
                print(f"Skipping non-UTF8 file: {pyfile}")

    print("Migration complete.")


if __name__ == "__main__":
    main()
