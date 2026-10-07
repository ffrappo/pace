"""Continue authentic BFCL memory prerequisites, binding saved target-made state."""
import json
import os
import subprocess
from pathlib import Path


def _digest(path):
    return subprocess.check_output(['b3sum', '--num-threads', '4', str(path)], text=True).split()[0]


def run_memory_prerequisites(handler, prerequisites, model_result_dir, model_name):
    from bfcl_eval.utils import get_directory_structure_by_id
    binding_path = Path(os.environ['PACE_NATIVE_BINDING'])
    if not binding_path.is_absolute() or not binding_path.is_file():
        raise RuntimeError('PACE_NATIVE_BINDING must name the real committed model/config binding')
    binding = _digest(binding_path)
    prepared, retained = [], []
    for entry in prerequisites:
        folder = model_result_dir / get_directory_structure_by_id(entry['id']) / 'memory_snapshot'
        snapshot = folder / 'prereq_checkpoints' / (entry['id'] + '.json')
        receipt = snapshot.with_suffix('.generation.json')
        latest = folder / (entry['scenario'] + '_final.json')
        # Load a fresh exact source entry before inference mutates its messages.
        source = json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        lineage = {'id': entry['id'], 'model': model_name,
                   'native_binding_blake3': binding, 'source_entry_json': source}
        if snapshot.exists() != receipt.exists():
            raise RuntimeError(f'Unpaired original BFCL memory snapshot/receipt: {snapshot}')
        if receipt.exists():
            if prepared:
                raise RuntimeError('Non-contiguous BFCL prerequisite continuation')
            record = json.loads(receipt.read_text())
            if record['lineage'] != lineage or record['snapshot_blake3'] != _digest(snapshot):
                raise RuntimeError(f'BFCL prerequisite state or lineage differs: {snapshot}')
            retained.append((record, latest))
        else:
            prepared.append((entry, snapshot, receipt, latest, lineage))
    if retained:
        record, latest = retained[-1]
        if not latest.is_file() or _digest(latest) != record['snapshot_blake3']:
            raise RuntimeError(f'Actual BFCL latest memory differs from completed prerequisite: {latest}')
    elif prepared and prepared[0][3].exists():
        raise RuntimeError('Unbound BFCL latest memory exists before original first prerequisite')
    for entry, snapshot, receipt, latest, lineage in prepared:
        responses, execution = handler.inference(entry, include_input_log=False, exclude_state_log=True)
        if not snapshot.is_file() or not latest.is_file() or _digest(snapshot) != _digest(latest):
            raise RuntimeError(f'Original BFCL prerequisite did not flush exact current memory: {snapshot}')
        with receipt.open('x') as file:
            json.dump({'lineage': lineage, 'snapshot_blake3': _digest(snapshot),
                       'model_result_raw': responses, 'metadata': execution}, file, default=str)
            file.flush()
            os.fsync(file.fileno())
        print(f"PACE_BFCL_MEMORY_PREREQ_COMPLETE model={model_name} id={entry['id']}", flush=True)
