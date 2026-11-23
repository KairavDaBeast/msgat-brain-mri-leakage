"""Turn a '# %%' script into a Jupyter notebook: python py2nb.py in.py out.ipynb"""
import json, sys
cells, cur, kind = [], [], "code"
def flush():
    src = "\n".join(cur).strip("\n")
    if src:
        if kind == "markdown":
            src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.splitlines())
        cells.append({"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
                      **({"outputs": [], "execution_count": None} if kind == "code" else {})})
for line in open(sys.argv[1], encoding="utf-8").read().splitlines():
    if line.startswith("# %%"):
        flush(); cur = []; kind = "markdown" if "[markdown]" in line else "code"
        if kind == "code" and line[4:].strip():
            cur.append("# " + line[4:].strip())
    else:
        cur.append(line)
flush()
json.dump({"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
           "nbformat": 4, "nbformat_minor": 5}, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print(len(cells), "cells")
