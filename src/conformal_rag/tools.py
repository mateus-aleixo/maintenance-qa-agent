"""Agent tools. Each tool: name, JSON-schema-ish description, run(args) -> str.

`predict_rul` calls the live conformal-rul API — the sibling project deployed on
AWS Lambda — so the agent composes two systems. Contract mirrors its OpenAPI:
POST /predict {subset, cycles: [{setting_1..3, s_01..s_21}], coverage, taxonomy}.
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .config import Config
from .embed import Embedder
from .guard import fence
from .retrieve import retrieve
from .store import Store


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable[[dict], str]


# -- search_docs -------------------------------------------------------------

def make_search_tool(store: Store, embedder: Embedder, cfg: Config) -> Tool:
    def run(args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: missing 'query'"
        hits = retrieve(store, embedder, query, cfg.k_bm25, cfg.k_vec, cfg.k_final, cfg.rrf_k)
        if not hits:
            return "no results"
        return "\n\n".join(
            f"[{i}] ({h.doc}, p.{h.page})\n{fence(h.text[:600])}"
            for i, h in enumerate(hits, start=1)
        )

    return Tool(
        name="search_docs",
        description='Search the maintenance manuals. Args: {"query": "<terms>"}',
        run=run,
    )


# -- calculator (safe AST eval, arithmetic only) -----------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


def make_calculator_tool() -> Tool:
    def run(args: dict) -> str:
        expr = str(args.get("expression", ""))
        if len(expr) > 200:
            return "error: expression too long"
        try:
            return str(_eval_node(ast.parse(expr, mode="eval")))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
            return f"error: {e}"

    return Tool(
        name="calculator",
        description='Arithmetic only. Args: {"expression": "3 * (2 + 1)"}',
        run=run,
    )


# -- predict_rul (live conformal-rul API) ------------------------------------

def make_rul_tool(cfg: Config, client: httpx.Client | None = None) -> Tool:
    http = client or httpx.Client(timeout=30)

    def run(args: dict) -> str:
        cycles = args.get("cycles")
        if not isinstance(cycles, list) or not cycles:
            return (
                "error: 'cycles' must be a non-empty list of readings, each with "
                "setting_1..setting_3 and s_01..s_21 as numbers"
            )
        payload = {
            "subset": args.get("subset", "FD001"),
            "cycles": cycles,
            "coverage": int(args.get("coverage", 90)),
            "taxonomy": args.get("taxonomy", "band"),
        }
        try:
            resp = http.post(f"{cfg.rul_api}/predict", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"error: RUL API unreachable ({e})"
        d = resp.json()
        return json.dumps(
            {
                "rul_cycles": d.get("rul_cycles"),
                "interval": d.get("interval"),
                "risk_band": d.get("risk_band"),
                "model": d.get("model"),
            }
        )

    return Tool(
        name="predict_rul",
        description=(
            "Predict remaining useful life of a turbofan engine with a calibrated "
            'interval, via the live conformal-rul service. Args: {"subset": "FD001", '
            '"cycles": [{"setting_1": 0.0, ..., "s_01": 518.67, ..., "s_21": 23.42}], '
            '"coverage": 90}'
        ),
        run=run,
    )
