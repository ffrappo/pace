"""Retain one failed model response after BFCL's original error transport broke."""
import json
import os
from pathlib import Path

from evaluations.generation import GenerationFailure, configure_generation, record_response
from evaluations.score_journal import ScoreJournal, record_digest


def main():
    root = Path('/root/eugeny-pace')
    native = root / 'native-journals/chatcmpl-b8709583-606b-4255-bdca-4454e6d71b27.jsonl'
    rows = [json.loads(line) for line in native.read_text().splitlines()]
    if rows[0]['event'] != 'request' or rows[-1]['event'] != 'response':
        raise RuntimeError('Actual terminal native request/response missing')
    response = rows[-1]['response']
    if response['id'] != rows[0]['id']:
        raise RuntimeError('Native terminal response identity differs')
    generation = configure_generation(root / 'generation.json', response['model'])
    os.environ['PACE_JOURNAL_CUSTODY'] = str(root / 'journal-custody.json')
    journal = ScoreJournal(root / 'score-journal', response['model'],
                           'http://127.0.0.1:8768/v1',
                           Path('/root/pace/scripts/pacebench/selections/abs_fit/selections_C100_text.csv'),
                           'Eugeny-Q2-Vast-20261012', generation)
    call = ['bfcl', 'agentic_memory_rec_sum', 'memory_rec_sum_45']
    try:
        prior = journal.history[-1]
        if prior['call'] != call or prior['status'] != 'failed' or prior.get('terminal'):
            raise RuntimeError('Original unretained BFCL failure is not the latest observation')
        if response['id'] not in prior['error']:
            raise RuntimeError('Actual native failed response absent from original error')
        try:
            record_response(response)
        except GenerationFailure as failure:
            if str(failure) != 'Explicit PACE generation has no final content or tool calls':
                raise
        else:
            raise RuntimeError('Saved response is not the original measured model failure')
        journal.append({'status': 'failed', 'call': call, 'terminal': True,
                        'elapsed_seconds': prior['elapsed_seconds'],
                        'error': 'Explicit PACE generation has no final content or tool calls',
                        'raw_result': response,
                        'raw_transport': [{'request': rows[0]['request']}, {'response': response}],
                        'native_journal': str(native),
                        'repairs_error_transport_of': record_digest(prior),
                        'repair': 'Retain authentic terminal model failure after SDK-history JSON serialization failed'})
        print('ACTUAL_FAILED_NATIVE_RESPONSE_RETAINED', response['id'], 'ungraded', flush=True)
    finally:
        journal.close()


if __name__ == '__main__':
    main()
