"""Append-only consumer correction through the owning BFCL CLI."""
from pathlib import Path

from evaluations.generation import (GenerationFailure, begin_transport, configure_generation,
                                    end_transport, record_response)
from evaluations.score_journal import ScoreJournal, record_digest


def repair_saved_claims(directory, profile_path, previous_digest, mode):
    """Withdraw affected saved claims; optionally invoke original BFCL grading only."""
    import json
    root = Path(directory)
    binding = json.loads((root / "binding.json").read_text())
    generation = configure_generation(profile_path, binding["model"])
    journal = ScoreJournal.for_replay(root, generation, previous_digest)
    try:
        candidates = {}
        for record in journal.history:
            if record["status"] != "scored":
                continue
            key = tuple(record["call"])
            responses = [item["response"] for item in record.get("raw_transport", []) if "response" in item]
            native_xml = (key[0] == "bfcl" and any(
                "<function=" in (response["choices"][0]["message"].get("content") or "")
                and not response["choices"][0]["message"].get("tool_calls") for response in responses))
            incomplete = any(response["choices"][0]["message"].get("incomplete_reasoning") is True
                             or (not (response["choices"][0]["message"].get("content") or "").strip()
                                 and not response["choices"][0]["message"].get("tool_calls"))
                             for response in responses)
            if native_xml or incomplete:
                candidates[key] = (record, native_xml, incomplete)
        corrected = []
        # Withdraw every defective claim before attempting any replay.
        for key, (record, native_xml, incomplete) in candidates.items():
            if journal.done(key) and journal.records[key]["status"] == "regraded":
                continue
            reason = "Native incomplete reasoning/final" if incomplete else "Native XML tool call was not parsed"
            original = journal.invalidate(key, reason)
            if record_digest(original) != record_digest(record):
                raise RuntimeError("Consumer repair candidate changed from original scored claim")
            if incomplete and not journal.pending[key].get("terminal"):
                token = begin_transport()
                try:
                    for item in original["raw_transport"]:
                        if "response" in item:
                            record_response(item["response"])
                    raise RuntimeError("Retained incomplete generation unexpectedly passed response contract")
                except GenerationFailure as exc:
                    journal.append({"status": "failed", "terminal": True, "call": list(key),
                                    "supersedes": record_digest(original), "error": str(exc),
                                    "raw_result": exc.raw_result, "raw_transport": original["raw_transport"]})
                finally:
                    end_transport(token)
            corrected.append({"call": list(key), "status": journal.pending[key]["status"]})
        repo = Path(__file__).resolve().parents[2]
        journal.reconcile_scores(repo / "results" / "standardized_results")
        if mode == "withdraw":
            return corrected
        from evaluations.handlers.bfcl import _run_bfcl
        for key, (record, native_xml, incomplete) in candidates.items():
            if incomplete or not native_xml or journal.done(key):
                continue
            token = begin_transport()
            try:
                result = _run_bfcl(binding["model"], binding["endpoint"], "", key[1], key[2],
                                   replay_record=record)
                transport = end_transport(token)
            except Exception as exc:
                transport = end_transport(token)
                journal.append({"status": "failed", "call": list(key), "replay_only": True,
                                "error": str(exc), "raw_result": getattr(exc, "raw_result", None),
                                "raw_transport": transport})
                raise
            if type(result[0].get("valid")) is not bool:
                raise RuntimeError("Original BFCL replay grader returned no boolean validity")
            scores = [{**score, "score": float(result[0]["valid"])} for score in record["scores"]]
            journal.append({"status": "regraded", "call": list(key), "supersedes": record_digest(record),
                            "scores": scores, "raw_result": result, "raw_transport": transport,
                            "original_raw_transport": record["raw_transport"], "replay_only": True})
            import csv
            for score in scores:
                path = repo / "results" / "standardized_results" / score["benchmark"] / score["subdir"]
                path.mkdir(parents=True, exist_ok=True)
                path /= binding["output_label"] + ".csv"
                rows = []
                if path.exists():
                    with path.open(newline="") as file:
                        rows = [row for row in csv.DictReader(file) if row["id"] != str(score["instance_id"])]
                rows.append({"id": score["instance_id"], "score": score["score"], "metric_name": score["metric"]})
                with path.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["id", "score", "metric_name"])
                    writer.writeheader(); writer.writerows(rows)
            corrected.append({"call": list(key), "status": "regraded", "scores": scores})
        return corrected
    finally:
        journal.close()
