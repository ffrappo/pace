"""Saved transport consumption for the real BFCL request owner, with no client."""
from copy import deepcopy
import json
import subprocess
from pathlib import Path

from evaluations.generation import record_response, request_configuration


class SavedBFCLResponses:
    def __init__(self, record):
        self.record = record
        self.responses = [item["response"] for item in record["raw_transport"] if "response" in item]
        self.requests = [item["request"] for item in record["raw_transport"] if "request" in item]
        if len(self.responses) != 1:
            raise ValueError("Single-turn BFCL replay requires exactly one saved response")
        if self.requests and len(self.requests) != len(self.responses):
            raise ValueError("Saved BFCL requests and responses differ in count")
        self.consumed = 0

    def next(self, kwargs):
        from openai.types.chat import ChatCompletion
        if self.consumed >= len(self.responses):
            raise RuntimeError("BFCL replay exhausted saved generations; network is unavailable in replay")
        request = request_configuration(kwargs)
        if self.requests and request != self.requests[self.consumed]:
            raise RuntimeError("Original BFCL replay prompt or tool schema differs from saved request")
        raw = self.responses[self.consumed]
        if raw["model"] != request["model"]:
            raise RuntimeError("Saved BFCL response belongs to another model")
        original = self.record["raw_result"][0]["model_result_raw"]
        if not self.requests and original != raw["choices"][0]["message"]["content"]:
            raise RuntimeError("Legacy BFCL journal output and saved native content differ")
        record_response(raw)
        self.consumed += 1
        return ChatCompletion.model_validate(deepcopy(raw)), self.record["raw_result"][0]["metadata"]["latency"]

    def complete(self):
        if self.consumed != len(self.responses):
            raise RuntimeError("BFCL replay did not consume every saved generation")


class SavedBFCLPrefix:
    """Consume an explicit completed native prefix; later queries stay with the live owner."""
    def __init__(self, reference):
        path = Path(reference['journal'])
        raw = path.read_bytes()
        digest = subprocess.check_output(['b3sum'], input=raw).decode().split()[0]
        if 'blake3:' + digest != reference['journal_digest'] or not raw.endswith(b'\n'):
            raise RuntimeError('BFCL native continuation journal digest/records differ')
        rows = [json.loads(line) for line in raw.splitlines()]
        if rows[0]['event'] != 'request' or rows[-1]['event'] != 'response':
            raise RuntimeError('BFCL continuation needs an authentic completed native response')
        self.request, self.response = rows[0]['request'], rows[-1]['response']
        if self.response['id'] != rows[0]['id'] or self.response['model'] != self.request['model']:
            raise RuntimeError('BFCL continuation response identity differs')
        self.latency = self.response['eugeny']['whole_request_seconds']
        self.reference = reference
        self.remaining = True

    def next(self, kwargs):
        from openai.types.chat import ChatCompletion
        from evaluations.generation import retain_transport
        if not self.remaining or request_configuration(kwargs) != self.request:
            raise RuntimeError('BFCL continuation original request/tool schema differs')
        record_response(self.response)
        retain_transport({'native_continuation': self.reference})
        self.remaining = False
        return ChatCompletion.model_validate(deepcopy(self.response)), self.latency

    def complete(self):
        if self.remaining:
            raise RuntimeError('BFCL continuation did not consume its original prefix')


if __name__ == "__main__":
    from evaluations._bfcl_runner import main
    main()
