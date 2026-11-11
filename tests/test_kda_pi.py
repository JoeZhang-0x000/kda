"""The pi driver of the KDA flow, against a stand-in for the pi binary."""
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kda_pi", ROOT / "scripts" / "kda_pi.py")
kda_pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kda_pi)

CONTRACT = {"name": "t", "objective": "be fast", "correctness": "exact", "target": "2x", "approaches": "python",
            "validation": "true", "evaluation": "true", "promotion": "faster and correct"}

# A pi that does what the prompt's phase asks: writes the file the driver waits for, and reports usage.
FAKE_PI = r'''#!/usr/bin/env python3
import json, os, re, sys, time
from pathlib import Path
prompt = sys.argv[-1]
Path(os.environ["FAKE_LOG"]).open("a").write(json.dumps(sys.argv[1:-1]) + "\n")
def reply(text, stop="stop"):
    print(json.dumps({"type": "tool_execution_start"}))
    print(json.dumps({"type": "message_end", "message": {"role": "assistant", "stopReason": stop,
          "content": [{"type": "text", "text": text}], "usage": {"input": 10, "output": 5, "totalTokens": 15}}}))
if "independent reviewer" in prompt:
    n = int(re.search(r"reviewer of round (\d+)", prompt).group(1))
    verdict = "COMPLETE" if n >= int(os.environ.get("FAKE_COMPLETE_AT", "1")) else "CONTINUE"
    Path(f".kda/round-{n}-review.md").write_text(f"looks fine\n{verdict}\n")
    reply("reviewed")
elif "round" in prompt and "implementation loop" in prompt:
    n = int(re.search(r"This is round (\d+)", prompt).group(1))
    if os.environ.get("FAKE_HANG") == str(n):
        time.sleep(60)
    Path("solution").mkdir(exist_ok=True)
    Path("solution/score").write_text(str(os.environ.get("FAKE_SCORES", "1,3,2").split(",")[n]))
    Path(f".kda/round-{n}-summary.md").write_text("did it")
    reply("implemented")
elif "docs/plan.md" in prompt and "Step 5" in prompt:
    Path("docs/plan.md").write_text("plan")
    reply("planned")
else:
    if os.environ.get("FAKE_NO_DRAFT"):
        reply("", stop="error")
    else:
        Path("docs/draft.md").write_text("draft")
        reply("drafted")
'''
EVAL = "import json, pathlib; p = pathlib.Path('solution/score'); " \
       "print(json.dumps({'correct': p.exists(), 'score': float(p.read_text()) if p.exists() else None}))"


@pytest.fixture
def env(tmp_path, monkeypatch):
    pi = tmp_path / "pi"
    pi.write_text(FAKE_PI)
    pi.chmod(pi.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("KDA_PI_BIN", str(pi))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "argv.jsonl"))
    (tmp_path / "contract.json").write_text(json.dumps(CONTRACT))
    (tmp_path / "eval.py").write_text(EVAL)
    return tmp_path


def run(env, *extra):
    ws = env / "ws"
    ws.mkdir(exist_ok=True)
    rc = kda_pi.main(["--workspace", str(ws), "--contract", str(env / "contract.json"), "--model", "a/impl",
                      "--reviewer-model", "b/rev", "--hours", "1",
                      "--eval-cmd", f"{sys.executable} {env / 'eval.py'}", *extra])
    read = lambda name: [json.loads(l) for l in (ws / ".kda" / name).read_text().splitlines()]
    return rc, ws, read


def test_contract_fills_every_placeholder():
    text = kda_pi.fill_contract(kda_pi.FLOW.read_text(), CONTRACT)
    assert "<fill in" not in text and "- Objective: be fast" in text and "## Workflow" in text
    with pytest.raises(SystemExit):
        kda_pi.fill_contract(kda_pi.FLOW.read_text(), {**CONTRACT, "validation": ""})


def test_flow_runs_draft_plan_rounds_and_stops_on_complete(env):
    rc, ws, read = run(env)
    assert rc == 0
    assert [u["label"] + "/" + u["role"] for u in read("usage.jsonl")] == [
        "draft/implementer", "plan/implementer", "round-0/implementer", "round-0/reviewer",
        "round-1/implementer", "round-1/reviewer"]
    assert all(u["ok"] and u["input"] == 10 and u["tool_calls"] == 1 for u in read("usage.jsonl"))
    assert json.loads((ws / ".kda" / "done.json").read_text())["stop"] == "complete"
    assert [s["score"] for s in read("snapshots.jsonl")] == [1.0, 3.0]
    assert (ws / ".kda" / "best" / "score").read_text() == "3"
    models = [a[a.index("--model") + 1] for a in map(json.loads, (env / "argv.jsonl").read_text().splitlines())]
    assert models == ["a/impl", "a/impl", "a/impl", "b/rev", "a/impl", "b/rev"]


def test_best_keeps_the_earlier_candidate_when_a_later_one_is_worse(env, monkeypatch):
    monkeypatch.setenv("FAKE_COMPLETE_AT", "2")
    rc, ws, read = run(env)
    assert [e["score"] for e in read("evals.jsonl")] == [None, 1.0, 3.0, 2.0]
    assert [s["label"] for s in read("snapshots.jsonl")] == ["round-0", "round-1"]
    assert (ws / ".kda" / "best" / "score").read_text() == "3"


def test_max_rounds_and_skills_reach_pi(env, monkeypatch):
    monkeypatch.setenv("FAKE_COMPLETE_AT", "99")
    skill = env / "wiki"
    skill.mkdir()
    rc, ws, read = run(env, "--max-rounds", "1", "--skill", str(skill), "--effort", "high")
    assert json.loads((ws / ".kda" / "done.json").read_text())["stop"] == "max_rounds"
    argv = json.loads((env / "argv.jsonl").read_text().splitlines()[0])
    assert argv[argv.index("--skill") + 1] == str(skill) and "--no-skills" not in argv
    assert argv[argv.index("--thinking") + 1] == "high"


def test_a_turn_that_times_out_still_leaves_a_usage_record(env, monkeypatch):
    monkeypatch.setenv("FAKE_HANG", "0")
    monkeypatch.setenv("FAKE_COMPLETE_AT", "0")
    rc, ws, read = run(env, "--turn-timeout", "2", "--max-rounds", "1")
    hung = [u for u in read("usage.jsonl") if u["label"] == "round-0" and u["role"] == "implementer"]
    assert rc == 0 and hung and hung[0]["timed_out"] and not hung[0]["ok"] and hung[0]["wall_s"] < 30


def test_no_draft_no_flow(env, monkeypatch):
    monkeypatch.setenv("FAKE_NO_DRAFT", "1")
    rc, ws, read = run(env)
    assert rc == 3 and not read("usage.jsonl")[0]["ok"]
