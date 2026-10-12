"""Durable observations for the existing PACE scorer, not an alternate grader."""
import hashlib
import json
import os
from pathlib import Path


def json_value(value):
    """Preserve authentic NumPy values; unsupported types remain errors."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise TypeError(f"NumPy dtype cannot be represented in JSON: {value.dtype}")
        return converted
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class ScoreJournal:
    def __init__(self, directory, model, endpoint, selection, output_label=None, generation=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.binding = {
            "model": model, "endpoint": endpoint,
            "selection_sha256": hashlib.sha256(Path(selection).read_bytes()).hexdigest(),
            "output_label": output_label or model, "generation": generation,
        }
        binding = self.root / "binding.json"
        if binding.exists():
            if json.loads(binding.read_text()) != self.binding:
                raise RuntimeError("PACE resume binding differs from the original run")
        else:
            binding.write_text(json.dumps(self.binding, indent=2) + "\n")
            binding.chmod(0o600)
        self.path = self.root / "observations.jsonl"
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                if record["status"] == "scored":
                    key = tuple(record["call"])
                    if key in self.records:
                        raise RuntimeError(f"Duplicate PACE observation: {key}")
                    self.records[key] = record
        self.log = self.path.open("a")
        self.path.chmod(0o600)

    def append(self, record):
        self.log.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_value) + "\n")
        self.log.flush()
        os.fsync(self.log.fileno())
        if record["status"] == "scored":
            self.records[tuple(record["call"])] = record

    def done(self, call):
        return tuple(call) in self.records

    def close(self):
        self.log.close()
