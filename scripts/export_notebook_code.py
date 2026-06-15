import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/nba_modeling.ipynb")
OUTPUT_PATH = Path("scripts/colab_export.py")


def main():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.strip():
                code_cells.append(f"# %% Cell {index}\n{source}\n\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Auto-generated code export from notebooks/nba_modeling.ipynb\n")
        f.write("# This file is used as a reference for refactoring notebook logic into src/.\n\n")
        f.writelines(code_cells)

    print(f"Exported {len(code_cells)} code cells to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
