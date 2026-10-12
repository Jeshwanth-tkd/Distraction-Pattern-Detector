"""
run_notebook.py
---------------
Tiny notebook executor (no Jupyter needed). Reads a "percent" script
(# %% code cells, # %% [markdown] cells), runs every code cell in one shared
namespace, captures printed text, displayed tables and matplotlib figures,
and writes a fully executed .ipynb that opens in Jupyter / VS Code / Colab / GitHub.

Usage: python src/run_notebook.py notebooks/distraction_project.py notebooks/Distraction_Detection.ipynb
"""
import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_cells(src):
    cells, cur, kind = [], [], None
    for line in src.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(cur).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(cur).strip("\n")))
    out = []
    for kind, body in cells:
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        if body.strip():
            out.append((kind, body))
    return out


def main(src_path, out_path):
    src_path = os.path.abspath(src_path)
    out_path = os.path.join(os.path.dirname(src_path), os.path.basename(out_path))
    os.chdir(os.path.dirname(src_path))
    cells = parse_cells(open(src_path, encoding="utf-8").read())
    ns = {"__name__": "__main__"}
    outputs_current = []

    def display(obj):
        data = {"text/plain": [str(obj)]}
        if hasattr(obj, "_repr_html_"):
            try:
                data["text/html"] = [obj._repr_html_()]
            except Exception:
                pass
        outputs_current.append({"output_type": "display_data", "data": data, "metadata": {}})

    ns["display"] = display
    nb_cells, n_exec = [], 0
    for kind, body in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)})
            continue
        n_exec += 1
        outputs_current.clear()
        buf = io.StringIO()
        try:
            tree = ast.parse(body)
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            if last is not None:
                tree.body = tree.body[:-1]
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {n_exec}>", "exec"), ns)
                result = eval(compile(ast.Expression(last.value), f"<cell {n_exec}>", "eval"), ns) if last else None
        except Exception:
            print(buf.getvalue())
            print(f"\n*** Error in cell {n_exec} ***\n{body[:400]}\n", file=sys.stderr)
            traceback.print_exc()
            if "results" in ns:
                import pandas as pd
                print(pd.DataFrame(ns["results"]).T.round(3).to_string(), file=sys.stderr)
            sys.exit(1)
        outs = []
        if buf.getvalue():
            outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(keepends=True)})
        outs.extend(outputs_current)
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=90, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(png.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        if result is not None:
            data = {"text/plain": [str(result)]}
            if hasattr(result, "_repr_html_"):
                data["text/html"] = [result._repr_html_()]
            outs.append({"output_type": "execute_result", "execution_count": n_exec, "data": data, "metadata": {}})
        nb_cells.append({"cell_type": "code", "execution_count": n_exec, "metadata": {},
                         "source": body.splitlines(keepends=True), "outputs": outs})
        print(f"cell {n_exec} ok", file=sys.stderr)

    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.abspath(out_path) if os.path.isabs(out_path) else os.path.join(os.getcwd(), os.path.basename(out_path)), "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out_path} ({len(nb_cells)} cells)", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
