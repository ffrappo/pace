"""Standardized CSV output and preservation of already committed journal scores."""
import csv
import sys
from collections import defaultdict
from contextlib import contextmanager

from evaluations.journal_custody import CustodyError


def flush_scores(root, output_label, written):
    """Merge accumulated triples into the standard CSVs, deduplicated by id."""
    for (benchmark, subdir), triples in written.items():
        if not triples:
            continue
        directory = root / benchmark / subdir if subdir else root / benchmark
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{output_label}.csv"
        existing = {}
        if path.exists():
            with path.open(newline="") as file:
                for row in csv.DictReader(file):
                    existing[row["id"]] = (row["score"], row["metric_name"])
        for iid, score, metric in triples:
            existing[str(iid)] = (score, metric)
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "score", "metric_name"])
            for iid, (score, metric) in existing.items():
                writer.writerow([iid, score, metric])


def flush_journal(root, journal):
    journal.reconcile_scores(root)
    restored = defaultdict(list)
    for record in journal.records.values():
        for score in record["scores"]:
            restored[(score["benchmark"], score["subdir"])].append(
                (score["instance_id"], score["score"], score["metric"]))
    flush_scores(root, journal.binding["output_label"], restored)


@contextmanager
def preserve_committed_scores(root):
    """Custody is not a model failure. Flush local committed scores, then rethrow."""
    try:
        yield
    except CustodyError as failure:
        if failure.journal is not None:
            journal = failure.journal
            try:
                flush_journal(root, journal)
            except Exception as exc:
                # Preserve the custody error even when the independent CSV write also fails.
                print(f"Committed PACE score flush also failed ({type(exc).__name__})", file=sys.stderr)
            finally:
                journal.close()
        raise
