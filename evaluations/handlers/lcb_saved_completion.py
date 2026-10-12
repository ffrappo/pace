"""Saved native completion consumption for actual LiveCodeBench, with no client."""
import json
from copy import deepcopy
from pathlib import Path

from evaluations.generation import GenerationFailure, record_response, request_configuration


class SavedNativeCompletion:
    """Consume the owning terminal HTTP envelope from one continued native journal."""

    def __init__(self, journal_path, source_digest, canonical_request_id):
        path = Path(journal_path)
        raw = path.read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError("Native continuation journal has an incomplete final line")
        rows = [json.loads(line) for line in raw.splitlines()]
        if not rows or rows[0]["event"] != "request" or rows[-1]["event"] != "response":
            raise ValueError("Native continuation journal lacks request/response envelope")
        if path.name != f"{canonical_request_id}.jsonl" or rows[0]["id"] != canonical_request_id:
            raise ValueError("Native continuation file belongs to another request")
        if ("blake3:" + __import__("subprocess").check_output(
                ["b3sum", str(path)], text=True).split()[0]) != source_digest:
            raise ValueError("Native continuation journal digest differs")
        response = rows[-1]["response"]
        if response["id"] != canonical_request_id or type(response["usage"]) is not dict:
            raise ValueError("Native completion belongs to another request")
        self.journal_path = str(path)
        self.request, self.response = rows[0]['request'], deepcopy(response)
        self.consumed = False

    def next(self, kwargs):
        if self.consumed:
            raise RuntimeError('Saved native completion already consumed; network unavailable')
        request = request_configuration(kwargs)
        if request['model'] != self.request['model'] or request['messages'] != self.request['messages']:
            raise ValueError('Original LiveCodeBench prompt/model differs from saved native request')
        for key in ('max_tokens', 'seed', 'stream', 'temperature', 'top_p', 'top_k',
                    'presence_penalty', 'repetition_penalty'):
            if request.get(key) != self.request.get(key):
                raise ValueError('Original LiveCodeBench sampling differs from saved native request')
        if request.get('tools') != self.request.get('tools'):
            raise ValueError('Original LiveCodeBench tool declaration differs from saved native request')
        usage = self.response["usage"]
        if (usage["prompt_tokens"] != len(self.prompt_tokens())
                or usage["completion_tokens"] != len(self.response["eugeny"]["output_tokens"])
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("Native completion usage differs from restored token state")
        if self.response["choices"][0]["finish_reason"] == "length":
            raise GenerationFailure("Incomplete PACE generation: finish_reason=length",
                                    self.response)
        record_response(self.response)
        self.consumed = True
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(deepcopy(self.response))

    def prompt_tokens(self):
        for line in Path(self.journal_path).read_text().splitlines():
            event = json.loads(line)
            if event['event'] == 'request':
                return event['prompt_tokens']
        raise ValueError('Native journal request tokens missing')

    def complete(self):
        if not self.consumed:
            raise RuntimeError("Saved native completion was not consumed")
