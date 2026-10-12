"""Consume an authentic terminal native envelope in the original LCB handler."""
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from evaluations.generation import (generation_config, record_response,
                                    request_configuration, retain_transport)


def load_saved_lcb(path):
    if path is None:
        return None
    record = json.loads(Path(path).read_bytes())
    if set(record) != {'call', 'native_continuation'}:
        raise ValueError('Saved LCB receipt requires exact call and native_continuation')
    call = record['call']
    if (not isinstance(call, list) or len(call) != 3 or call[:2] != ['livecodebench', 'codegeneration']
            or not isinstance(call[2], str) or not call[2].isdigit()):
        raise ValueError('Saved native completion requires one LCB codegeneration call')
    native = record['native_continuation']
    if set(native) != {'journal', 'journal_digest', 'request_id'}:
        raise ValueError('Saved LCB receipt requires exact journal, digest and request ID')
    if not Path(native['journal']).is_absolute() or not native['journal_digest'].startswith('blake3:'):
        raise ValueError('Saved LCB receipt requires absolute journal and BLAKE3 digest')
    return record


class SavedNativeCompletion:
    """No client, renderer, sampling or replacement completion is created here."""

    def __init__(self, journal_path, source_digest, canonical_request_id):
        path = Path(journal_path)
        raw = path.read_bytes()
        if not raw.endswith(b'\n'):
            raise ValueError('Native continuation journal has an incomplete final line')
        digest = subprocess.check_output(['b3sum'], input=raw).decode().split()[0]
        if 'blake3:' + digest != source_digest:
            raise ValueError('Native continuation journal digest differs')
        rows = [json.loads(line) for line in raw.splitlines()]
        if (not rows or rows[0]['event'] != 'request' or rows[-1]['event'] != 'response'
                or sum(row['event'] == 'request' for row in rows) != 1
                or sum(row['event'] == 'response' for row in rows) != 1):
            raise ValueError('Native continuation requires one request and final response')
        origin, response = rows[0], rows[-1]['response']
        if path.name != f'{canonical_request_id}.jsonl' or origin['id'] != canonical_request_id:
            raise ValueError('Native continuation file belongs to another request')
        if response['id'] != canonical_request_id or response['model'] != origin['request']['model']:
            raise ValueError('Native completion belongs to another request or model')
        self.prompt, self.profile = origin['prompt_tokens'], origin['profile']
        retired, last_accounting = [], None
        for row in rows[1:-1]:
            reply = row['reply']
            if row['event'] == 'native_init' and not retired:
                retired.append(reply['anchor'])
            elif row['event'] == 'native_step' and retired:
                if reply['emitted'] != [reply['anchor']]:
                    raise ValueError('Native continuation requires ordinary one-token retired steps')
                retired += reply['emitted']
            else:
                raise ValueError('Native continuation journal event ordering differs')
            last_accounting = reply['accounting']
            if (last_accounting['prompt_tokens'] != len(self.prompt)
                    or last_accounting['generated_tokens'] != len(retired)
                    or last_accounting['target_draws'] != len(retired)
                    or last_accounting['committed_length'] != len(self.prompt) + len(retired) - 1):
                raise ValueError('Native continuation retired step accounting differs')
        if not retired or response['eugeny']['output_tokens'] != retired:
            raise ValueError('Native completion output differs from actual retired journal IDs')
        final_accounting = response['eugeny']['accounting']
        for key in ('schema', 'capacity', 'prompt_tokens', 'generated_tokens', 'target_draws',
                    'committed_length', 'max_new_tokens', 'output', 'stop'):
            if final_accounting[key] != last_accounting[key]:
                raise ValueError('Native completion accounting differs from last retired step')
        self.request, self.response = origin['request'], deepcopy(response)
        self.provenance = {'path': str(path), 'blake3': source_digest,
                           'request_id': canonical_request_id, 'mode': 'saved_native_completion'}
        self.consumed = False

    def next(self, kwargs):
        if self.consumed:
            raise RuntimeError('Saved native completion already consumed; network unavailable')
        config = generation_config()
        if config is None or self.profile != config['profile']:
            raise ValueError('Saved native completion requires matching explicit generation profile')
        request = request_configuration(kwargs)
        request.pop('timeout', None)  # Client read timeout is not an outgoing native field.
        if request != self.request:
            raise ValueError('Original LiveCodeBench prompt/model/sampling differs from saved native request')
        usage, native = self.response['usage'], self.response['eugeny']
        tokens, accounting = native['output_tokens'], native['accounting']
        if (not isinstance(self.prompt, list) or not self.prompt or not isinstance(tokens, list)
                or not tokens or any(type(token) is not int for token in self.prompt + tokens)):
            raise ValueError('Native completion lacks literal original token IDs')
        if (usage['prompt_tokens'] != len(self.prompt) or usage['completion_tokens'] != len(tokens)
                or usage['total_tokens'] != len(self.prompt) + len(tokens)
                or accounting['prompt_tokens'] != len(self.prompt)
                or accounting['generated_tokens'] != len(tokens)
                or accounting['target_draws'] != len(tokens)
                or accounting['max_new_tokens'] != request['max_tokens']
                or accounting['capacity'] != config['context_tokens']):
            raise ValueError('Native completion usage differs from restored token state')
        choices = self.response['choices']
        if len(choices) != 1 or choices[0]['finish_reason'] not in ('stop', 'length'):
            raise ValueError('Native completion has invalid final choice')
        terminal = choices[0]['finish_reason'] == 'stop'
        if terminal != (accounting['stop'] == 'model_terminal'):
            raise ValueError('Native completion finish reason differs from native accounting')
        retain_transport({'request': self.request, 'native_continuation': self.provenance})
        # Retain the raw length/incomplete response before the common contract rejects it.
        record_response(self.response)
        self.consumed = True
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(deepcopy(self.response))

    def complete(self):
        if not self.consumed:
            raise RuntimeError('Saved native completion was not consumed')
