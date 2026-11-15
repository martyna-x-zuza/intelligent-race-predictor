"""Every function in src/ - public or private - documents its parameters and result.

ruff's pydocstyle rules check public functions only, so this test covers the private ones too.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "race_predictor"

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _functions() -> list[tuple[str, FunctionNode]]:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                found.append((f"{path.relative_to(SRC)}:{node.lineno} {node.name}", node))
    return found


def _parameters(node: FunctionNode) -> list[str]:
    args = node.args
    names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
    names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
    return [name for name in names if name not in {"self", "cls"}]


def _returns_something(node: FunctionNode) -> bool:
    return not (node.returns is None or ast.unparse(node.returns) == "None")


FUNCTIONS = _functions()


@pytest.mark.parametrize("node", [n for _, n in FUNCTIONS], ids=[i for i, _ in FUNCTIONS])
def test_function_is_documented(node: FunctionNode) -> None:
    docstring = ast.get_docstring(node)
    assert docstring, "missing docstring"
    if _parameters(node):
        assert "Args:" in docstring, "parameters are not documented"
    if _returns_something(node):
        assert "Returns:" in docstring or "Yields:" in docstring, "result is not documented"


def test_the_check_sees_the_whole_package() -> None:
    assert len(FUNCTIONS) > 80
