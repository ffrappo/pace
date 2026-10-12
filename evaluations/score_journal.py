"""Durable observations for the existing PACE scorer, not an alternate grader."""
import hashlib
import json
import os
from pathlib import Path

from evaluations.journal_custody import CustodyError, JournalCustody


def json_value(value):
    """Preserve authentic NumPy values and native path metadata; unknown types fail."""
    if isinstance(value, Path):
        return str(value)
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise TypeError(f"NumPy dtype cannot be represented in JSON: {value.dtype}")
        return converted
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def record_digest(record):
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False, default=json_value)
    return hashlib.sha256(encoded.encode()).hexdigest()


class ScoreJournal:
    def __init__(self, directory, model, endpoint, selection, output_label=None, generation=None,
                 migrate_endpoint_from=None):
        binding = {"model": model, "endpoint": endpoint,
                   "selection_sha256": hashlib.sha256(Path(selection).read_bytes()).hexdigest(),
                   "output_label": output_label or model, "generation": generation}
        self._open(directory, binding, migrate_endpoint_from)

    @classmethod
    def for_replay(cls, directory, generation, migrate_parser_from=None):
        journal = cls.__new__(cls)
        journal._open(directory, None)
        expected = {**journal.binding, "generation": generation}
        if journal.binding != expected:
            if migrate_parser_from is None:
                journal.close()
                raise RuntimeError("Replay needs explicit --migrate-parser-from for changed generation binding")
            journal.migrate_parser(expected, migrate_parser_from)
        return journal

    def _open(self, directory, expected, migrate_endpoint_from=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        binding_path = self.root / "binding.json"
        if binding_path.exists():
            self._binding_bytes = binding_path.read_bytes()
            self.binding = json.loads(self._binding_bytes)
        else:
            if expected is None or migrate_endpoint_from is not None:
                raise RuntimeError("Replay or endpoint migration requires an existing journal binding")
            if (self.root / "observations.jsonl").exists():
                raise RuntimeError("PACE observations require their original journal binding")
            self.binding = expected
            self._binding_bytes = (json.dumps(self.binding, indent=2) + "\n").encode("utf-8")
            with binding_path.open("wb") as file:
                file.write(self._binding_bytes)
                file.flush()
                os.fsync(file.fileno())
            binding_path.chmod(0o600)
        self.path = self.root / "observations.jsonl"
        self.records, self.pending, self.history, self.by_digest = {}, {}, [], {}
        self._observations_bytes = self.path.read_bytes() if self.path.exists() else b""
        if self._observations_bytes and not self._observations_bytes.endswith(b"\n"):
            raise RuntimeError("PACE journal has an incomplete final line")
        for line in self._observations_bytes.splitlines():
            self._apply(json.loads(line))
        if migrate_endpoint_from is not None and self.binding["endpoint"] != migrate_endpoint_from:
            raise RuntimeError("Explicit endpoint migration does not match original endpoint")
        migrate = expected is not None and self.binding != expected
        if migrate and (migrate_endpoint_from is None or {
                k: v for k, v in self.binding.items() if k != "endpoint"} != {
                k: v for k, v in expected.items() if k != "endpoint"}):
            raise RuntimeError("PACE resume binding differs from the original run")
        self.log = self.path.open("a", encoding="utf-8", newline="")
        self.log.flush()
        os.fsync(self.log.fileno())
        self.path.chmod(0o600)
        self.custody, self._custody_error = None, None
        try:
            self.custody = JournalCustody.from_env()
            self.sync_custody()
            if migrate:
                self.append({"status": "binding_migrated", "reason": "Explicit endpoint migration",
                             "previous_binding": self.binding, "binding": expected})
        except CustodyError as exc:
            exc.journal = self
            self._custody_error = exc
            self.close()
            raise

    def sync_custody(self):
        if self._custody_error is not None:
            raise self._custody_error
        if self.custody is None:
            return
        try:
            if ((self.root / "binding.json").read_bytes() != self._binding_bytes
                    or self.path.read_bytes() != self._observations_bytes):
                raise CustodyError("PACE custody local journal bytes changed outside committed state")
            self.custody.publish(self._binding_bytes, self._observations_bytes)
        except (CustodyError, OSError) as exc:
            failure = exc if isinstance(exc, CustodyError) else CustodyError(
                f"PACE custody local read failure ({type(exc).__name__})")
            failure.journal = self
            self._custody_error = failure
            raise failure from None

    def _apply(self, record, validate_only=False):
        status = record["status"]
        if status == "binding_migrated":
            if record["previous_binding"] != self.binding:
                raise RuntimeError("PACE migration does not bind the previous state")
            if not validate_only:
                self.binding = record["binding"]
        elif status in {"scored", "regraded"}:
            key = tuple(record["call"])
            if key in self.records:
                raise RuntimeError(f"Duplicate active PACE observation: {key}")
            if status == "regraded":
                if key not in self.pending or record["supersedes"] != self.pending[key]["supersedes"]:
                    raise RuntimeError("PACE regrade lacks exact prior withdrawal")
                if not validate_only:
                    del self.pending[key]
            elif key in self.pending:
                raise RuntimeError("Withdrawn PACE generation requires replay, not regeneration")
            if not validate_only:
                self.records[key] = record
        elif status == "invalidated":
            key = tuple(record["call"])
            if key not in self.records or record_digest(self.records[key]) != record["supersedes"]:
                raise RuntimeError("PACE invalidation does not bind the active claim")
            if not validate_only:
                del self.records[key]
                self.pending[key] = record
        elif status == "failed" and record.get("terminal"):
            key = tuple(record["call"])
            if key in self.records:
                raise RuntimeError("Terminal PACE failure cannot override an active score")
            if "supersedes" in record:
                if key not in self.pending or record["supersedes"] != self.pending[key]["supersedes"]:
                    raise RuntimeError("Terminal PACE failure lacks prior withdrawal")
            elif key in self.pending:
                raise RuntimeError("Repeated terminal failure must bind original observation")
            if not validate_only:
                self.pending[key] = record
        elif status != "failed":
            raise RuntimeError(f"Unknown PACE journal status: {status}")
        if not validate_only:
            self.history.append(record)
            self.by_digest[record_digest(record)] = record

    def append(self, record):
        if self._custody_error is not None:
            raise self._custody_error
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_value) + "\n"
        self._apply(record, validate_only=True)
        self.log.write(encoded)
        self.log.flush()
        os.fsync(self.log.fileno())
        self._apply(record)
        self._observations_bytes += encoded.encode("utf-8")
        self.sync_custody()

    def migrate_parser(self, expected, previous_digest):
        from evaluations.generation import generation_config
        old, new = self.binding["generation"], expected["generation"]
        if old["sha256"] != previous_digest or old.get("tool_parser") is not None:
            raise RuntimeError("Explicit parser migration does not match original generation digest")
        if new.get("tool_parser") != "qwen3_coder":
            raise RuntimeError("Only qwen3_coder parser migration is supported")
        if {k: v for k, v in old.items() if k not in {"sha256", "tool_parser"}} != {
                k: v for k, v in new.items() if k not in {"sha256", "tool_parser"}}:
            raise RuntimeError("Parser migration changed original generation provenance")
        config = dict(generation_config())
        del config["tool_parser"]
        prior_bytes = (json.dumps(config, indent=2) + "\n").encode()
        if hashlib.sha256(prior_bytes).hexdigest() != previous_digest:
            raise RuntimeError("Parser migration changed more than the tool_parser setting")
        if {k: v for k, v in expected.items() if k != "generation"} != {
                k: v for k, v in self.binding.items() if k != "generation"}:
            raise RuntimeError("Parser migration changed scorer binding")
        self.append({"status": "binding_migrated", "reason": "Explicit Qwen3 Coder response parser repair",
                     "previous_binding": self.binding, "binding": expected})

    def invalidate(self, call, reason):
        key = tuple(call)
        if key in self.pending:
            return self.by_digest[self.pending[key]["supersedes"]]
        original = self.records[key]
        self.append({"status": "invalidated", "call": list(key), "reason": reason,
                     "supersedes": record_digest(original)})
        return original

    def reconcile_scores(self, standardized_root):
        """Remove only withdrawn claims from this journal's original output label."""
        import csv
        groups = {}
        for record in self.pending.values():
            if "supersedes" not in record:
                continue
            original = self.by_digest[record["supersedes"]]
            for score in original["scores"]:
                key = (score["benchmark"], score["subdir"])
                groups.setdefault(key, set()).add(str(score["instance_id"]))
        for (benchmark, subdir), ids in groups.items():
            path = Path(standardized_root) / benchmark
            if subdir:
                path /= subdir
            path /= self.binding["output_label"] + ".csv"
            if not path.exists():
                continue
            with path.open(newline="") as file:
                reader = csv.DictReader(file)
                columns, rows = reader.fieldnames, list(reader)
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(row for row in rows if row["id"] not in ids)

    def done(self, call):
        return tuple(call) in self.records

    def blocked(self, call):
        return tuple(call) in self.pending

    def terminal(self, call):
        return self.pending.get(tuple(call), {}).get("terminal", False)

    def close(self):
        self.log.close()
