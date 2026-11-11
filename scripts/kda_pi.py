#!/usr/bin/env python3
"""The KDA flow, driven on pi (https://github.com/badlogic/pi-mono) with no human in it.

    kda_pi.py --workspace WS --contract contract.json --model provider/id --reviewer-model provider/id
              --hours 4 [--eval-cmd "bash evaluate.sh"] [--solution-dir solution] [--skill DIR ...]

Upstream KDA is a prompt (prompts/basic-flow.md), two skills, and the Humanize plugin, which runs the
implementation as an RLCR loop: one agent implements a round, a second, independent agent reviews it, and
the review is the next round's input. Humanize does that with Claude Code / Codex Stop hooks. pi has no
such hook, so this file is that loop:

  draft    the implementer gets the filled-in basic-flow prompt and writes docs/draft.md       (flow steps 1-4)
  plan     the implementer turns the draft into docs/plan.md                                  (step 5; gen-plan --direct)
  round N  the implementer implements one candidate, validates it and writes round-N-summary.md (steps 6-8)
           the reviewer checks the round against the plan and writes round-N-review.md, whose last line
           is COMPLETE or CONTINUE
  ...      until the reviewer says COMPLETE, --max-rounds (Humanize's default, 42), or the time budget.

Left out, on purpose: Humanize's plan-understanding quiz and its questions to the user (nobody is there),
its git-branch and push checks, and its finalize phase.

Every turn is a fresh pi session; state lives in the workspace. The driver commits the workspace after
every turn. With --eval-cmd it also runs the task's evaluation itself after each implementer turn: the
command's last stdout line must be a JSON object with "correct" (bool) and "score" (number, higher is
better). The best correct --solution-dir is kept in .kda/best/, and .kda/snapshots.jsonl says at how many
seconds each one was reached, so a run can be read at any budget. .kda/usage.jsonl has one line per turn:
tokens, cost, provider retries and the time spent waiting on them, wall time, and whether it timed out.

The time budget is --hours from the start; a turn is cut to what is left of it. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOW = HERE.parent / "prompts" / "basic-flow.md"
EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
CONTRACT = (("name", "Task name"), ("objective", "Objective"), ("correctness", "Correctness requirements"),
            ("target", "Performance or quality target"), ("approaches", "Allowed implementation approaches"),
            ("validation", "Validation command"), ("evaluation", "Evaluation command"),
            ("promotion", "Promotion criteria"))


def fill_contract(flow: str, contract: dict) -> str:
    """basic-flow.md with every `<fill in ...>` of the task contract replaced."""
    missing = [k for k, _ in CONTRACT if not str(contract.get(k, "")).strip()]
    if missing:
        raise SystemExit(f"kda_pi: the contract has no {missing}")
    out = []
    for line in flow.splitlines():
        for key, label in CONTRACT:
            if line.startswith(f"- {label}:"):
                line = f"- {label}: {contract[key]}"
        out.append(line)
    return "\n".join(out) + "\n"


class Turn:
    """What pi's event stream says about one turn."""

    def __init__(self) -> None:
        self.text = ""
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0, "cost": 0.0}
        self.messages = self.tool_calls = self.retries = 0
        self.retry_wait_s = 0.0
        self.errors: list[str] = []
        self.failed = False
        self._retry_started: float | None = None

    def observe(self, e: dict, now: float) -> None:
        kind = e.get("type")
        if kind == "message_end" and (e.get("message") or {}).get("role") == "assistant":
            m = e["message"]
            u = m.get("usage") or {}
            for ours, theirs in (("input", "input"), ("output", "output"), ("cache_read", "cacheRead"),
                                 ("cache_write", "cacheWrite"), ("total", "totalTokens")):
                self.usage[ours] += u.get(theirs) or 0
            self.usage["cost"] += (u.get("cost") or {}).get("total") or 0.0
            self.messages += 1
            text = "".join(c.get("text", "") for c in m.get("content") or [] if c.get("type") == "text")
            if text.strip():
                self.text = text
            self.failed = m.get("stopReason") in ("error", "aborted")
            if m.get("stopReason") == "error":
                self.errors.append(str(m.get("errorMessage") or "provider error"))
        elif kind == "tool_execution_start":
            self.tool_calls += 1
        elif kind == "auto_retry_start":
            self.retries += 1
            self._retry_started = now
        elif kind == "auto_retry_end":
            if self._retry_started is not None:
                self.retry_wait_s += now - self._retry_started
                self._retry_started = None
            if not e.get("success"):
                self.failed = True
                self.errors.append(str(e.get("finalError") or "retry failed"))


def pi_argv(model: str, effort: str, sessions: Path, skills: list[str], prompt: str) -> list[str]:
    argv = [os.environ.get("KDA_PI_BIN", "pi"), "--print", "--mode", "json", "--session-dir", str(sessions),
            "--no-context-files", "--no-extensions", "--no-prompt-templates", "--no-themes", "--no-mcp",
            "--no-approve", "--offline", "--model", model]
    if effort:
        argv += ["--thinking", effort]
    for s in skills:
        argv += ["--skill", s]
    if not skills:
        argv.append("--no-skills")
    return [*argv, "--", prompt]


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.a = args
        self.ws = Path(args.workspace).resolve()
        self.state = self.ws / ".kda"
        self.state.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.deadline = self.t0 + args.hours * 3600
        self.best: float | None = None
        self.contract = fill_contract(FLOW.read_text(), json.loads(Path(args.contract).read_text()))
        self.skills = [str(Path(s).resolve()) for s in args.skill]

    # -- plumbing ---------------------------------------------------------------------------------
    def say(self, msg: str) -> None:
        print(f"[kda {int(time.time() - self.t0):6d}s] {msg}", flush=True)

    def left(self) -> float:
        return self.deadline - time.time()

    def log(self, name: str, record: dict) -> None:
        with (self.state / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def git(self, *argv: str) -> str:
        return subprocess.run(["git", *argv], cwd=self.ws, capture_output=True, text=True).stdout.strip()

    def commit(self, msg: str) -> str:
        self.git("add", "-A")
        self.git("-c", "user.name=kda-pi", "-c", "user.email=kda-pi@localhost", "commit", "-q", "--allow-empty",
                 "-m", msg)
        return self.git("rev-parse", "--short", "HEAD")

    def prepare(self) -> None:
        if not (self.ws / ".git").exists():
            self.git("init", "-q")
        exclude = self.ws / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        if ".kda/sessions/" not in (exclude.read_text() if exclude.exists() else ""):
            with exclude.open("a") as f:
                f.write(".kda/sessions/\n.kda/best/\n")
        (self.ws / "docs").mkdir(exist_ok=True)
        self.commit("kda: start")

    # -- one agent turn ---------------------------------------------------------------------------
    def turn(self, role: str, label: str, prompt: str) -> bool:
        """Run one pi turn; True when it ended with a reply. Always leaves a usage record."""
        model, effort = (self.a.model, self.a.effort) if role == "implementer" else \
                        (self.a.reviewer_model, self.a.reviewer_effort)
        if self.left() < 60:
            return False
        limit = min(self.a.turn_timeout, self.left())
        self.say(f"{label}: {role} turn on {model} (up to {int(limit)} s)")
        t, turn = time.time(), Turn()
        proc = subprocess.Popen(pi_argv(model, effort, self.state / "sessions", self.skills, prompt), cwd=self.ws,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(limit, kill)
        timer.start()
        err: list[bytes] = []
        drain = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
        drain.start()
        for raw in proc.stdout:                                  # pi frames events with LF
            try:
                turn.observe(json.loads(raw.decode("utf-8")), time.time())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        rc = proc.wait()
        timer.cancel()
        drain.join(5)
        ok = rc == 0 and not turn.failed and bool(turn.text.strip()) and not timed_out.is_set()
        self.log("usage.jsonl", {"at_s": round(t - self.t0, 1), "label": label, "role": role, "model": model,
                                 "effort": effort, "wall_s": round(time.time() - t, 1), "rc": rc, "ok": ok,
                                 "timed_out": timed_out.is_set(), **turn.usage, "assistant_messages": turn.messages,
                                 "tool_calls": turn.tool_calls, "retries": turn.retries,
                                 "retry_wait_s": round(turn.retry_wait_s, 1), "errors": turn.errors[-3:]})
        (self.state / f"{label}-{role}-reply.md").write_text(turn.text)
        if not ok:
            why = "timed out" if timed_out.is_set() else "; ".join(turn.errors[-3:]) or f"pi exited with {rc}"
            self.say(f"{label}: {role} turn did not finish ({why}) {b''.join(err)[-300:].decode('utf-8', 'replace')}")
        return ok

    # -- the driver's own measurement -------------------------------------------------------------
    def evaluate(self, label: str, commit: str) -> dict | None:
        if not self.a.eval_cmd:
            return None
        t = time.time()
        try:
            p = subprocess.run(shlex.split(self.a.eval_cmd), cwd=self.ws, capture_output=True, text=True,
                               timeout=self.a.eval_timeout)
            out, rc = p.stdout, p.returncode
        except subprocess.TimeoutExpired:
            out, rc = "", 124
        result: dict = {"correct": False, "score": None}
        for line in reversed(out.strip().splitlines()):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict) and "correct" in doc:
                result = doc
                break
        record = {"at_s": round(time.time() - self.t0, 1), "label": label, "commit": commit, "rc": rc,
                  "eval_s": round(time.time() - t, 1), **result}
        self.log("evals.jsonl", record)
        score = result.get("score")
        self.say(f"{label}: evaluation correct={result.get('correct')} score={score}")
        if result.get("correct") and isinstance(score, (int, float)) and (self.best is None or score > self.best):
            self.best = float(score)
            src, dst = self.ws / self.a.solution_dir, self.state / "best"
            if src.is_dir():
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            self.log("snapshots.jsonl", {**record, "best": True})
            self.say(f"{label}: new best {score}")
        return record

    # -- the flow ---------------------------------------------------------------------------------
    def budget_note(self) -> str:
        return (f"\nTime: about {max(0, int(self.left() / 60))} minutes of the run's budget are left. Nobody is "
                "watching this session: do not ask questions, decide and proceed.\n")

    def run(self) -> int:
        self.prepare()
        self.evaluate("start", self.commit("kda: start"))
        draft = (self.contract + self.budget_note() +
                 "\nDo steps 1-4 of the workflow now and stop: write the implementation-plan draft to "
                 "`docs/draft.md`. Do not implement anything yet.\n")
        if not (self.ws / "docs" / "draft.md").exists():
            self.turn("implementer", "draft", draft)
            self.commit("kda: draft")
        if not (self.ws / "docs" / "draft.md").exists():
            self.say("no docs/draft.md was written; the flow does not start without it")
            return 3
        if not (self.ws / "docs" / "plan.md").exists():
            self.turn("implementer", "plan", self.contract + self.budget_note() +
                      "\nStep 5: turn `docs/draft.md` into an executable plan in `docs/plan.md`. Order the "
                      "candidates, and give each one its concrete steps, the exact validation and evaluation "
                      "commands, and the evidence that promotes or rejects it. Do not implement anything yet.\n")
            self.commit("kda: plan")
        plan = "docs/plan.md" if (self.ws / "docs" / "plan.md").exists() else "docs/draft.md"
        stop = "budget"
        for n in range(self.a.max_rounds):
            if self.left() < 120:
                break
            label = f"round-{n}"
            review = f".kda/round-{n - 1}-review.md"
            self.turn("implementer", label, self.contract + self.budget_note() +
                      f"\nThis is round {n} of the implementation loop (workflow steps 6-8). The plan is `{plan}`."
                      + (f" The independent review of the previous round is `{review}`: address it first.\n"
                         if n and (self.ws / review).exists() else "\n") +
                      f"Implement one candidate, keep the deliverable in `{self.a.solution_dir}/`, and run the "
                      "validation command. A candidate that is not better than the current one, or not correct, "
                      "must not stay in the deliverable. Record it in `candidates.jsonl` (name, parent, status, "
                      "reason) and its measurements in `benchmark.csv`. Then write "
                      f"`.kda/{label}-summary.md`: what you changed, the commands you ran and their results, and "
                      "what should be tried next. Stop after the summary.\n")
            commit = self.commit(f"kda: {label}")
            ev = self.evaluate(label, commit)
            if self.left() < 120:
                break
            ok = self.turn("reviewer", label, self.contract + self.budget_note() +
                           f"\nYou are the independent reviewer of round {n}; another agent did the work. Read "
                           f"`{plan}`, `.kda/{label}-summary.md`, the round's diff (`git show --stat HEAD`, `git "
                           "show HEAD`) and the evidence files. Check the summary's claims against what is in the "
                           "workspace; rerun the validation command if a claim is not supported by it."
                           + (f" The driver's own evaluation of this round: {json.dumps(ev)}." if ev else "") +
                           f" Do not edit `{self.a.solution_dir}/` or any implementation file. Write "
                           f"`.kda/{label}-review.md`: what is wrong or unproven, and the most valuable next "
                           "candidate from the plan. Its last line must be exactly `COMPLETE` if the plan is "
                           "carried out and nothing in it is still worth trying, otherwise exactly `CONTINUE`.\n")
            self.commit(f"kda: {label} review")
            verdict = ""
            if ok and (self.ws / f".kda/{label}-review.md").exists():
                lines = (self.ws / f".kda/{label}-review.md").read_text().strip().splitlines()
                verdict = lines[-1].strip().strip("`*") if lines else ""
            self.say(f"{label}: review verdict {verdict or '(none)'}")
            if verdict == "COMPLETE":
                stop = "complete"
                break
        else:
            stop = "max_rounds"
        summary = {"stop": stop, "elapsed_s": round(time.time() - self.t0, 1), "best_score": self.best}
        (self.state / "done.json").write_text(json.dumps(summary) + "\n")
        self.commit("kda: done")
        self.say(f"done: {json.dumps(summary)}")
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workspace", required=True, help="the task workspace; the agents work here")
    ap.add_argument("--contract", required=True,
                    help="JSON with " + ", ".join(k for k, _ in CONTRACT) + " (the task contract of basic-flow.md)")
    ap.add_argument("--model", required=True, help='the implementer, "provider/id" as pi names it')
    ap.add_argument("--effort", default="", choices=("", *EFFORTS))
    ap.add_argument("--reviewer-model", required=True, help="the independent reviewer (Humanize: Codex)")
    ap.add_argument("--reviewer-effort", default="", choices=("", *EFFORTS))
    ap.add_argument("--hours", type=float, required=True, help="wall-clock budget of the whole run")
    ap.add_argument("--turn-timeout", type=float, default=5400, help="seconds; Humanize's --codex-timeout default")
    ap.add_argument("--max-rounds", type=int, default=42, help="Humanize's default")
    ap.add_argument("--skill", action="append", default=[], help="a skill directory for both agents; repeatable")
    ap.add_argument("--solution-dir", default="solution", help="the deliverable, relative to the workspace")
    ap.add_argument("--eval-cmd", default="", help="run by the driver after each round; see the module docstring")
    ap.add_argument("--eval-timeout", type=float, default=1800)
    return Run(ap.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
