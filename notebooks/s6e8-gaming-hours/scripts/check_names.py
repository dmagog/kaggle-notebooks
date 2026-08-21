"""Catch a name the notebook uses but never defines, before Kaggle spends 85 minutes finding it.

Version 29 died on `NameError: name 'TOP10_GAPS_2026_08_09' is not defined`. The constant had been
renamed in two places out of three, the builder ran clean locally because the builder only writes
cells, and the notebook compiled fine because a NameError is a runtime error, not a syntax one.

This walks the cells in execution order, collects every name that becomes bound as it goes
(assignments, imports, defs, classes, comprehension and loop targets, with-items, except-names,
function arguments, walrus), and reports any name loaded before anything binds it.

It over-reports rather than under-reports, so the useful output is the diff against a known-clean
run rather than an empty list: attribute chains on late-bound globals, names introduced by
`display`-style helpers and anything created dynamically will show up. What it does catch reliably
is the thing that actually broke: a rename that survived in one place.

Run:  python3 check_names.py [notebook.ipynb]
"""
import ast
import builtins
import json
import sys


def _params(a):
    """Every parameter name an arguments node binds."""
    return {arg.arg for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
            if arg is not None}


def bound_by(node):
    """Every name this statement binds, at any depth inside it."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out |= _params(n.args)
        elif isinstance(n, ast.Lambda):
            # A lambda binds its parameters too. Missing this reported `lambda d: d.x` as an
            # unbound `d`, which is the kind of false alarm that gets a checker ignored.
            out |= _params(n.args)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out.update(n.names)
    return out


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "s6e8_notebook.ipynb"
    nb = json.load(open(path))
    known = set(dir(builtins)) | {"__name__", "__file__", "get_ipython", "display"}
    problems = []

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append((i, f"SYNTAX: {e}"))
            continue
        # A cell can use a name it binds later in the same cell (inside a function body that
        # only runs afterwards), so bind the whole cell first, then check loads against the
        # names known BEFORE it plus its own.
        cell_binds = set()
        for stmt in tree.body:
            cell_binds |= bound_by(stmt)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in known and n.id not in cell_binds:
                    problems.append((i, f"uses `{n.id}` (line {n.lineno} of the cell)"))
        known |= cell_binds

    seen, uniq = set(), []
    for i, msg in problems:
        if msg not in seen:
            seen.add(msg)
            uniq.append((i, msg))

    # Clobber check: a function def'd in one cell, re-bound by a later cell (loop target,
    # plain assignment), then loaded later still. v32 died exactly this way: a loop variable
    # named `key` overwrote the legend-handle helper thirty cells before a figure called it.
    clobbers = []
    func_def_cell, last_bind_kind = {}, {}
    cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
             if c["cell_type"] == "code"]
    parsed = []
    for i, src in cells:
        try: parsed.append((i, ast.parse(src)))
        except SyntaxError: continue
    for i, tree in parsed:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_def_cell.setdefault(node.name, i)
    for i, tree in parsed:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                f = func_def_cell.get(node.id)
                if f is not None and i > f:
                    used_later = any(isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load)
                                     and m.id == node.id
                                     for j, t2 in parsed if j > i for m in ast.walk(t2))
                    if used_later:
                        clobbers.append((i, f"rebinds `{node.id}` (a function from cell {f}) "
                                            f"and a later cell still loads it"))
    for i, msg in clobbers:
        if (i, msg) not in [(a_, b_) for a_, b_ in uniq]:
            uniq.append((i, msg))

    if not uniq:
        print(f"{path}: every loaded name is bound by an earlier cell, and no helper is clobbered. clean.")
    else:
        print(f"{path}: {len(uniq)} name(s) loaded before anything binds them\n")
        for i, msg in uniq:
            print(f"  cell {i:>3}  {msg}")
        print("\nSome of these may be false alarms; a name that is genuinely never assigned "
              "anywhere in the file is not.")
    sys.exit(1 if uniq else 0)


if __name__ == "__main__":
    main()
