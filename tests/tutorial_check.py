"""Run every Python code block of docs/TUTORIAL.md (use case 1 only up to the layer selection, which needs no model).

    python tests/tutorial_check.py
"""
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    text = (ROOT / "docs" / "TUTORIAL.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) == 4, len(blocks)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)                      # use case 1 writes my_layers.txt
        try:
            ns = {}
            exec(blocks[0], ns); exec(blocks[1], ns)            # use case 1: layer selection and layer file
            assert Path("my_layers.txt").exists()
            exec(blocks[2], {}); print("PASS use case 2")       # deployable layer in a custom model
            exec(blocks[3], {}); print("PASS use case 3")       # switching diagnostic
        finally:
            os.chdir(cwd)
    print("ALL PASS")


if __name__ == "__main__":
    main()
