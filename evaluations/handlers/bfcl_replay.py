"""Saved transport consumption for the real BFCL request owner, with no client."""
from copy import deepcopy

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


if __name__ == "__main__":
    from evaluations._bfcl_runner import main
    main()
