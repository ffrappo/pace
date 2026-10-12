"""Grade the original saved DebugBench output after its credential-locality failure."""
import json
import os
from pathlib import Path
import urllib.request

from evaluations.generation import configure_generation
from evaluations.score_journal import ScoreJournal, record_digest
from evaluations.score_new_model import extract


def main():
    root = Path('/root/eugeny-pace')
    journal_dir = root / 'score-journal'
    records = [json.loads(line) for line in (journal_dir / 'observations.jsonl').read_text().splitlines()]
    prior = records[-1]
    if prior['status'] != 'failed' or prior['call'][0] != 'debugbench' or prior.get('terminal'):
        raise RuntimeError('Saved original DebugBench grading failure is not pending')
    raw = prior['raw_result']
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0]['fixed_code'], str):
        raise RuntimeError('Original DebugBench fixed code missing; no regeneration permitted')
    result = raw[0]
    if result['test_result_bool'] is not None or result['test_result_dict'] is not None:
        raise RuntimeError('Original DebugBench result was already graded')
    responses = [item['response'] for item in prior['raw_transport'] if 'response' in item]
    if len(responses) != 1 or responses[0]['choices'][0]['finish_reason'] != 'stop':
        raise RuntimeError('Actual original completed neural response missing')
    original = responses[0]['choices'][0]['message']['content']
    expected = original[original.find('<code>') + len('<code>'):original.find('</code>')]
    if expected != result['fixed_code']:
        raise RuntimeError('Saved original DebugBench code differs from original parser output')
    request = {'code': result['fixed_code'], 'language': str(prior['call'][2]).split('_', 1)[0],
               'task_id': result['slug']}
    token = (root / 'debugbench-grader.token').read_text().strip()
    query = urllib.request.Request('http://127.0.0.1:8771/debugbench', data=json.dumps(request).encode(),
                                   headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(query, timeout=180) as response:
        graded = json.load(response)
    if graded['request'] != request or graded['grader'] != 'original_debugbench_leetcode_oj.LeetCodeTester':
        raise RuntimeError('Actual original grader input or owner differs')
    result['test_result_bool'], result['test_result_dict'] = graded['test_result_bool'], graded['test_result_dict']
    metric, value = extract('debugbench', None, raw)
    if value is None:
        raise RuntimeError('Actual original DebugBench grader returned no score')
    os.environ['PACE_JOURNAL_CUSTODY'] = str(root / 'journal-custody.json')
    generation = configure_generation(root / 'generation.json', responses[0]['model'])
    journal = ScoreJournal(journal_dir, responses[0]['model'], 'http://127.0.0.1:8768/v1',
                           Path('/root/pace/scripts/pacebench/selections/abs_fit/selections_C100_text.csv'),
                           'Eugeny-Q2-Vast-20261012', generation)
    try:
        journal.append({'status': 'scored', 'call': prior['call'],
                        'scores': [{'benchmark': 'debugbench', 'subdir': '', 'instance_id': prior['call'][2],
                                    'metric': metric, 'score': value}],
                        'raw_result': raw, 'raw_transport': prior['raw_transport'] + [{'original_debugbench_grader': graded}],
                        'elapsed_seconds': prior['elapsed_seconds'], 'repairs_grading_of': record_digest(prior)})
        print('ORIGINAL_SAVED_DEBUGBENCH_GRADED', prior['call'][2], metric, value, flush=True)
    finally:
        journal.close()


if __name__ == '__main__':
    main()
