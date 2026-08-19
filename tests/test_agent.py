import json

import httpx

from conformal_rag.agent import Agent
from conformal_rag.llm import StubLLM
from conformal_rag.tools import make_calculator_tool, make_rul_tool, make_search_tool


def test_agent_tool_then_final(store, embedder, cfg):
    llm = StubLLM(script=[
        json.dumps({"tool": "search_docs", "args": {"query": "generator starting checks"}}),
        json.dumps(
            {"final": "Inspect fuel lines, battery electrolyte, ground "
                      "[tm-generator.pdf p.4]."}
        ),
    ])
    agent = Agent.build(llm, [make_search_tool(store, embedder, cfg), make_calculator_tool()])
    run = agent.run("What do I check before starting the generator?")
    assert run.stopped == "final"
    assert len(run.steps) == 1
    assert run.steps[0].tool == "search_docs"
    assert "fuel lines" in run.steps[0].observation
    assert run.final.startswith("Inspect")


def test_agent_recovers_from_malformed_json_once():
    llm = StubLLM(script=["not json at all", json.dumps({"final": "ok"})])
    agent = Agent.build(llm, [make_calculator_tool()])
    run = agent.run("q")
    assert run.stopped == "final" and run.final == "ok"


def test_agent_gives_up_after_two_malformed():
    llm = StubLLM(script=["garbage", "still garbage"])
    agent = Agent.build(llm, [make_calculator_tool()])
    run = agent.run("q")
    assert run.stopped == "malformed" and run.final is None


def test_agent_unknown_tool_reports_and_continues():
    llm = StubLLM(script=[
        json.dumps({"tool": "rm_rf", "args": {}}),
        json.dumps({"final": "done"}),
    ])
    agent = Agent.build(llm, [make_calculator_tool()])
    run = agent.run("q")
    assert "unknown tool" in run.steps[0].observation
    assert run.final == "done"


def test_agent_max_steps_cap():
    call = json.dumps({"tool": "calculator", "args": {"expression": "1+1"}})
    llm = StubLLM(script=[call])  # repeats forever
    agent = Agent.build(llm, [make_calculator_tool()], max_steps=3)
    run = agent.run("q")
    assert run.stopped == "max-steps"
    assert len(run.steps) == 3


def test_calculator_is_arithmetic_only():
    calc = make_calculator_tool()
    assert calc.run({"expression": "3 * (2 + 1)"}) == "9"
    assert "error" in calc.run({"expression": "__import__('os').system('dir')"})
    assert "error" in calc.run({"expression": "().__class__"})
    assert "error" in calc.run({"expression": "1/0"})


def test_rul_tool_calls_live_contract(cfg):
    """MockTransport pins the request contract to the real conformal-rul API."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "rul_cycles": 42.5,
            "interval": {"lower": 31.0, "upper": 55.0, "coverage": 90},
            "risk_band": "warning",
            "model": "lgbm-fd001",
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = make_rul_tool(cfg, client=client)
    cycles = [{"setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
               **{f"s_{i:02d}": 1.0 for i in range(1, 22)}}]
    out = json.loads(tool.run({"cycles": cycles, "coverage": 90}))

    assert seen["url"].endswith("/predict")
    assert seen["body"]["subset"] == "FD001"
    assert seen["body"]["coverage"] == 90
    assert set(seen["body"]["cycles"][0]) == {"setting_1", "setting_2", "setting_3",
                                              *{f"s_{i:02d}" for i in range(1, 22)}}
    assert out["risk_band"] == "warning"
    assert out["interval"]["lower"] == 31.0


def test_rul_tool_rejects_missing_cycles(cfg):
    tool = make_rul_tool(cfg, client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))
    ))
    assert tool.run({}).startswith("error:")
