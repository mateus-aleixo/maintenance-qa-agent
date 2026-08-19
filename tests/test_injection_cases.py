"""Red-team cases from evals/injection_cases.jsonl, run in CI on every push.

Pattern flags are a first line, not the defence: end-to-end behaviour under a
real model is measured in `make eval` (M4). But the deterministic layer must not
regress silently, and benign controls must not be over-flagged.
"""

import json
from pathlib import Path

import pytest

from conformal_rag.guard import fence, flag

CASES = [
    json.loads(line)
    for line in (Path(__file__).parent.parent / "evals" / "injection_cases.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case(case):
    report = flag(case["payload"])
    if case["expect"] == "flagged":
        assert report.suspicious, f"{case['id']}: should be flagged, was not"
    elif case["expect"] == "clean":
        assert not report.suspicious, f"{case['id']}: false positive {report.flags}"
    elif case["expect"] == "defused":
        inner = (
            fence(case["payload"])
            .removeprefix("<untrusted-data>\n")
            .removesuffix("\n</untrusted-data>")
        )
        assert "</untrusted-data>" not in inner
    else:
        pytest.fail(f"unknown expectation {case['expect']!r}")
