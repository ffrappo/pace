#!/usr/bin/env python3
"""Score a NEW model on the selected source instances and write standardized CSVs.

Closes the loop between `evaluations/` and `pacebench/`:

    selections CSV  --(this script: run each instance, extract score)-->
        results/standardized_results/<benchmark>/<subdir>/<NewModel>.csv
    --> scripts/pacebench/analysis/predict_new_model.py

For every row in the selection CSV it calls the benchmark handler once per unique
(benchmark, subtask, instance_id), extracts the metric that matches the row's
subdir, and appends `(instance_id, score, metric_name)` to the standardized CSV
that pacebench reads.

Usage
-----
  # Credentials come from the environment. Either export them for the session:
  export API_KEY=...   # OpenAI-compatible endpoint that serves the new model
  export BASE_URL=...
  # ...or (recommended) put them once in a git-ignored repo-root .env file:
  #   API_KEY=sk-...
  #   BASE_URL=https://cmu.litellm.ai
  # Real env vars / --flags override .env.
  python evaluations/score_new_model.py \
      --model azure_ai/gpt-5.2 \
      --model-name-out My-New-Model \
      --base-url https://cmu.litellm.ai \
      --selections scripts/pacebench/selections/abs_fit/selections_C100.csv

`--model` is the id sent to the API; `--model-name-out` is the filename written
under standardized_results (defaults to `--model`). Re-running resumes: instances
already present in the output CSVs are skipped.

Support / setup notes (all use the benchmark's authentic grading)
-----------------------------------------------------------------
- Most benchmarks score in-process in the main env.
- bfcl: scored in its own env (evaluations/.bfcl-venv, created by install.sh) via
  a subprocess — the main env can't import BFCL's tree-sitter build.
- acp_gen: grading needs the acpbench deps pinned in requirements-main.txt
  (lark / tarski[clingo] / pddl / kstar-planner / clingo).
- planbench: task_3 is a binary string match; all OTHER tasks are graded with the
  original VAL pipeline and require $VAL to be set (see
  benchmarks/planbench/plan-bench/install_val.py, and _validate_planbench.py to
  confirm the grades reproduce the standardized results).
- debugbench: needs LeetCode cookies in keys.json.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from evaluations.run import run_instance  # noqa: E402
from evaluations._env import load_env  # noqa: E402
from evaluations.score_journal import ScoreJournal  # noqa: E402
from evaluations.journal_custody import CustodyError  # noqa: E402
from evaluations.handlers.lcb_saved_completion import load_saved_lcb  # noqa: E402
from evaluations.score_output import flush_scores, flush_journal, preserve_committed_scores  # noqa: E402
from evaluations.generation import (configure_generation, begin_transport, end_transport,
                                    retain_transport, GenerationFailure, terminal_generation_kind)  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_STD = _REPO / "results" / "standardized_results"
_REPOBENCH_METRIC = re.compile(r"_(codebleu|edit_similarity|exact_match)$")
# BFCL needs its own virtualenv (tree-sitter 0.21.x hard-conflicts with the main
# env's 0.23.x). install.sh step [2/2] creates it here; we score BFCL instances by
# running evaluations/_bfcl_runner.py under this interpreter as a subprocess.
_BFCL_VENV_PY = _REPO / "evaluations" / ".bfcl-venv" / "bin" / "python"
_BFCL_RUNNER = _REPO / "evaluations" / "_bfcl_runner.py"
SUBDIR_IS_SUBTASK = {"planbench", "bfcl", "livecodebench", "lifbench", "acp_gen",
                     "visualwebbench"}
METRIC_SUBDIR = {"ifeval", "logiqa", "gpqa", "aime25", "mmlu_cot",
                 "humaneval_chat", "mbpp_chat"}
# Handlers that are NOT thread-safe (os.chdir, lm_eval global harness state,
# os.environ mutation) — serialized behind a lock; everything else runs concurrently.
SERIAL_BENCHMARKS = {"livecodebench", "bfcl", "debugbench",
                     "acp_gen", "aime25", "gpqa", "humaneval_chat", "ifeval",
                     "logiqa", "mbpp_chat", "mmlu_cot"}


def _num(v):
    """bool/None/number -> float or None."""
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    if v is None:
        return None
    return float(v)


def eval_call(benchmark, subdir, instance_id):
    """(benchmark, subdir, id) -> (subtask_or_None, instance_id) for run_instance."""
    if benchmark == "repobench":
        return _REPOBENCH_METRIC.sub("", subdir), instance_id
    if benchmark in METRIC_SUBDIR:
        return None, instance_id
    if benchmark in SUBDIR_IS_SUBTASK and subdir:
        return subdir, instance_id
    return None, instance_id


def extract(benchmark, subdir, result):
    """Return (metric_name, score) for this row's subdir, or None if unscorable."""
    r = result[0] if isinstance(result, list) and result else result
    if not isinstance(r, dict):
        return None
    b = benchmark
    if b == "visualpuzzles":
        return ("exact_match", _num(r.get("correct")))
    if b == "mmmu":
        return ("mmmu_acc", _num(r.get("correct")))
    if b == "visualwebbench":
        # metric_name varies by task (rouge_l / f1 / correct); the handler emits
        # the normalized [0,1] score under '<metric>_normalized'.
        for mk in ("rouge_l", "f1", "correct"):
            if f"{mk}_normalized" in r:
                return (mk, _num(r[f"{mk}_normalized"]))
        return None
    if b == 'livecodebench':
        value = r.get('pass@1', r.get('pass_at_1'))
        if value is None:
            raise RuntimeError('Authentic LiveCodeBench pass@1 missing')
        return ('pass@1', _num(value))
    if b == "bfcl":
        return ("accuracy", _num(r.get("valid")))
    if b == 'lifbench':
        value = (r.get('score_dict') or {}).get('total_score')
        if value is None:
            raise RuntimeError('Authentic LIFBench total_score missing')
        return ('total_score', _num(value))
    if b == 'logiqa':
        if r.get('exact_match') is None:
            raise RuntimeError('Authentic LogiQA exact_match missing')
        return ('exact_match', _num(r['exact_match']))
    if b == 'acp_gen':
        if r.get('score') is None:
            raise RuntimeError('Authentic ACP score missing')
        return ('score', _num(r['score']))
    if b in ("aime25", "gpqa", "mmlu_cot"):
        return ("exact_match", _num(r.get("exact_match", 0.0)))
    if b == "humaneval_chat":
        return ("pass@1", _num(r.get("pass@1", r.get("pass_at_1", 0.0))))
    if b == "mbpp_chat":
        return ("pass_at_1", _num(r.get("pass_at_1", r.get("pass@1", 0.0))))
    if b == "infobench":
        return ("accuracy", _num((r.get("accuracy_percent") or 0.0) / 100.0))
    if b == "debugbench":
        return ("debug_accuracy", _num(r.get("test_result_bool")))
    if b == "repobench":
        for metric in ("codebleu", "edit_similarity", "exact_match"):
            if subdir.endswith("_" + metric):
                value = r.get(metric)
                if value is None:
                    raise RuntimeError(f'Authentic RepoBench {metric} missing')
                return (metric, _num(value))
        return None
    if b == "ifeval":
        v = r.get(subdir)
        if isinstance(v, list):
            v = (sum(1.0 for x in v if x) / len(v)) if v else 0.0
        return (subdir, _num(v))
    if b == "planbench":
        # task_3 -> 'llm_correct_binary'; all other tasks are graded with the
        # authentic VAL pipeline (needs $VAL set) and stored under 'llm_correct'.
        # Both metric_names match the standardized CSVs.
        if subdir == "task_3_plan_verification":
            return None if r.get("llm_correct_binary") is None else \
                ("llm_correct_binary", _num(r["llm_correct_binary"]))
        return None if r.get("llm_correct") is None else \
            ("llm_correct", _num(r["llm_correct"]))
    return None


def _load_existing(path):
    if not path.exists():
        return set()
    return {row["id"] for row in csv.DictReader(open(path))}


def _run_bfcl_subprocess(model, base_url, api_key, subtask, instance_id):
    """Score one BFCL instance in the dedicated .bfcl-venv, returning run_instance's shape.

    BFCL can't be scored in-process here: its tree-sitter pin conflicts with the
    main env (fails with "No module named 'tree_sitter_javascript'"). We instead
    run evaluations/_bfcl_runner.py under evaluations/.bfcl-venv/bin/python, which
    calls the real `_run_bfcl` and prints the [result] list as JSON. The returned
    list is the same shape run_instance would produce, so extract() is unchanged.

    Credentials are passed on the subprocess **stdin** (JSON) so they never appear
    in argv / `ps` output or on disk.
    """
    if not _BFCL_VENV_PY.exists():
        raise RuntimeError(
            f"BFCL virtualenv interpreter not found at {_BFCL_VENV_PY}. "
            f"Create it with `bash evaluations/install.sh` (step [2/2])."
        )
    payload = json.dumps({
        "model_name": model,
        "base_url": base_url,
        "api_key": api_key,
        "subtask": subtask,
        "instance_id": str(instance_id),
    })
    proc = subprocess.run(
        [str(_BFCL_VENV_PY), str(_BFCL_RUNNER)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(_REPO),            # repo root -> `evaluations` package importable
        env=os.environ.copy(),     # inherit HF_* etc.; api_key still comes via stdin
    )
    if proc.returncode != 0:
        detail = proc.stdout.strip()
        raw_result = None
        try:                       # runner reports failures as a JSON error object
            err = json.loads(detail)
            raw_result = err.get("raw_result")
            for record in err.get("raw_transport", []):
                retain_transport(record)
            detail = err.get("error") or err.get("traceback") or detail
        except (ValueError, TypeError):
            detail = detail or proc.stderr.strip()
        raise GenerationFailure(f"bfcl subprocess failed: {detail}", raw_result)
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(
            f"bfcl subprocess produced no output (stderr: {proc.stderr.strip()[-500:]})"
        )
    result = json.loads(out)
    if isinstance(result, dict) and "raw_transport" in result:
        for record in result["raw_transport"]:
            retain_transport(record)
        return result["result"]
    return result


@preserve_committed_scores(_STD)
def main():
    load_env()  # populate env from repo-root .env (real env vars still win)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selections", required=True)
    ap.add_argument("--model", required=True, help="model id sent to the API")
    ap.add_argument("--model-name-out", default=None,
                    help="filename under standardized_results (default: --model)")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://cmu.litellm.ai"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY"))
    ap.add_argument("--limit", type=int, default=0, help="only process the first N rows (debug)")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent API calls (default 8; unsafe handlers stay serialized)")
    ap.add_argument("--journal-dir", type=Path, help="durable raw observations and resume binding")
    ap.add_argument("--migrate-endpoint-from", help="explicit original URL for endpoint-only journal migration")
    ap.add_argument("--generation-config", default=os.environ.get("PACE_GENERATION_CONFIG"),
                    help="opt-in generation JSON, inherited by BFCL subprocesses")
    ap.add_argument('--saved-lcb-completion', type=Path, help='exact native receipt for one restored LCB call')
    args = ap.parse_args()
    saved_lcb = load_saved_lcb(args.saved_lcb_completion)
    if not args.api_key:
        ap.error("set --api-key or export API_KEY")
    out_name = args.model_name_out or args.model
    generation = configure_generation(args.generation_config, args.model)
    if generation is not None and args.journal_dir is None:
        ap.error("explicit generation requires --journal-dir for bound raw results and resume")
    if (os.environ.get("PACE_JOURNAL_CUSTODY") is not None or args.migrate_endpoint_from is not None
            ) and args.journal_dir is None:
        ap.error("custody and endpoint migration require --journal-dir")

    rows = list(csv.DictReader(open(args.selections)))
    if args.limit:
        rows = rows[:args.limit]

    def flush(written):
        flush_scores(_STD, out_name, written)

    # skip instances already scored in a previous run (resume)
    def already_done(b, subdir, iid):
        out_csv = (_STD / b / subdir if subdir else _STD / b) / f"{out_name}.csv"
        return out_csv.exists() and str(iid) in _load_existing(out_csv)

    journal = ScoreJournal(args.journal_dir, args.model, args.base_url, args.selections,
                           out_name, generation, args.migrate_endpoint_from) \
        if args.journal_dir is not None else None
    if journal is not None:
        flush_journal(_STD, journal)

    # Group rows by unique API call (dedup: repobench's metric rows share one call).
    rows_by_key = defaultdict(list)
    failed = set()
    for row in rows:
        b, subdir, iid = row["benchmark"], row["subdir"], row["instance_id"]
        subtask, call_iid = eval_call(b, subdir, iid)
        key = (b, subtask, call_iid)
        if journal is not None and journal.terminal(key):
            failed.add(key)
            continue
        if journal is not None and journal.blocked(key):
            raise RuntimeError(f"Withdrawn PACE call requires saved replay: {key}")
        if journal.done(key) if journal is not None else already_done(b, subdir, iid):
            continue
        rows_by_key[key].append((b, subdir, iid))

    if saved_lcb and tuple(saved_lcb['call']) not in rows_by_key:
        raise ValueError('Saved LCB receipt is not a pending selected canonical call')
    serial_lock = threading.Lock()

    def _call(b, subtask, call_iid):
        # BFCL can't run in-process (tree-sitter version conflict) -> score it in
        # its own venv via a subprocess; everything else runs in-process.
        if b == "bfcl":
            return _run_bfcl_subprocess(
                model=args.model, base_url=args.base_url, api_key=args.api_key,
                subtask=subtask, instance_id=call_iid)
        return run_instance(model_name=args.model, base_url=args.base_url,
                            api_key=args.api_key, benchmark=b, subtask=subtask, instance_id=call_iid,
                            replay_record=saved_lcb if saved_lcb and saved_lcb['call'] == [b, subtask, call_iid] else None)

    def run_one(key):
        b, subtask, call_iid = key
        started = time.monotonic()
        token = begin_transport()
        try:
            if b in SERIAL_BENCHMARKS:   # not thread-safe (chdir / global state / env)
                with serial_lock:
                    r = _call(b, subtask, call_iid)
            else:
                r = _call(b, subtask, call_iid)
            return key, r, None, time.monotonic() - started, end_transport(token)
        except CustodyError:
            end_transport(token)
            raise
        except Exception as e:
            return key, getattr(e, "raw_result", None), e, time.monotonic() - started, end_transport(token)
    written = defaultdict(list)   # (benchmark, subdir) -> [(id, score, metric)]
    total = len(rows_by_key)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # No queued model call may start before the prior append's custody GET succeeds.
        serial = args.workers == 1 or (journal is not None and journal.custody is not None)
        futs = [] if serial else [pool.submit(run_one, k) for k in rows_by_key]
        completed = (run_one(k) for k in rows_by_key) if serial else (f.result() for f in as_completed(futs))
        for key, result, err, elapsed, transport in completed:
            done += 1
            scores = []
            try:
                if err is not None:
                    raise err
                if result is None:
                    raise RuntimeError(f'PACE handler returned no result for {key}')
                for (bb, subdir, iid) in rows_by_key[key]:
                    ex = extract(bb, subdir, result)
                    if ex is None or ex[1] is None:
                        raise RuntimeError(f'PACE grader returned no grade for {(bb, subdir, iid)}')
                    metric, score = ex
                    scores.append(dict(benchmark=bb, subdir=subdir, instance_id=iid,
                                       metric=metric, score=score))
            except CustodyError:
                raise
            except Exception as failure:
                kind = terminal_generation_kind(failure, result, transport)
                terminal = kind is not None
                if journal is not None:
                    journal.append(dict(status='failed', call=key, elapsed_seconds=elapsed,
                                        error=str(failure), raw_result=result, raw_transport=transport,
                                        terminal=terminal, failure_kind=kind))
                print(f"[{done}/{total}] {key} FAILED: {failure}", file=sys.stderr, flush=True)
                if terminal:
                    failed.add(key)
                    continue
                if journal is not None: journal.close()
                for pending in futs: pending.cancel()
                flush(written)
                raise
            if journal is not None:
                journal.append(dict(status='scored', call=key, elapsed_seconds=elapsed,
                                    scores=scores, raw_result=result, raw_transport=transport))
            for s in scores:
                written[(s['benchmark'], s['subdir'])].append(
                    (s['instance_id'], s['score'], s['metric']))
                print(f"[{done}/{total}] {s['benchmark']}/{s['subdir']}/{s['instance_id']} -> "
                      f"{s['metric']}={s['score']}", flush=True)
            flush(written)
            written = defaultdict(list)
    if journal is not None: journal.close()
    if failed:
        raise RuntimeError(f"PACE retains {len(failed)} terminal response/capability failures; result remains unqualified")
    print(f"Wrote scores to CSVs under {_STD}/<benchmark>/.../{out_name}.csv")


if __name__ == "__main__":
    main()
