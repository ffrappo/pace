"""Opt-in request settings for the existing PACE producers, never a renderer."""
import hashlib
from copy import deepcopy
import json
import os
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

_CONFIG_ENV = "PACE_GENERATION_CONFIG"
_DIGEST_ENV = "PACE_GENERATION_CONFIG_SHA256"
_TRANSPORT = ContextVar("pace_raw_transport", default=None)
_SDK_FIELDS = {"model", "max_tokens", "seed", "stream", "temperature", "top_p",
               "presence_penalty"}
_GENERATION_FIELDS = {"max_tokens", "max_completion_tokens", "max_output_tokens",
                      "max_gen_toks", "temperature", "top_p", "top_k", "min_p",
                      "presence_penalty", "repetition_penalty", "frequency_penalty",
                      "seed", "stream", "do_sample", "reasoning_effort", "thinking",
                      "enable_thinking", "preserve_thinking", "chat_template_kwargs"}


class GenerationFailure(RuntimeError):
    def __init__(self, message, raw_result=None):
        super().__init__(message)
        self.raw_result = raw_result


def terminal_generation_kind(failure, raw, transport):
    """Classify retained native responses and the exact fixed-capacity refusal only."""
    if not isinstance(failure, GenerationFailure) or not isinstance(raw, dict):
        return None
    if (any(item.get("response") == raw for item in transport)
            and not any("error" in item or "status_code" in item for item in [raw, *transport])):
        return "model_response"
    config = generation_config()
    if (config is None or raw.get("type") != "BadRequestError"
            or raw.get("status_code") != 400 or raw not in transport):
        return None
    try:
        body = json.loads(raw["body"])
    except (KeyError, TypeError, ValueError):
        return None
    expected = {"error": {"message": "Rendered prompt plus budget exceeds context; no clipping"}}
    requests = [item["request"] for item in transport if "request" in item]
    if body != expected or not requests:
        return None
    request = requests[-1]
    if (request.get("model") != config["request"]["model"]
            or request.get("max_tokens") != config["request"]["max_tokens"]):
        return None
    return "context_capacity"


def begin_transport():
    return _TRANSPORT.set([])


def end_transport(token):
    records = _TRANSPORT.get()
    _TRANSPORT.reset(token)
    return records


def retain_transport(record):
    records = _TRANSPORT.get()
    if records is not None:
        records.append(deepcopy(record))


@lru_cache(maxsize=8)
def _load(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError("PACE generation config changed during the run")
    config = json.loads(raw)
    if config["schema"] != "pace.generation.v1":
        raise ValueError("Unsupported PACE generation config schema")
    request = config["request"]
    required = {"model", "max_tokens", "seed", "stream"}
    allowed = _SDK_FIELDS | {"top_k", "repetition_penalty"}
    if not required <= request.keys() or request.keys() - allowed:
        raise ValueError("PACE generation request has missing or unsupported fields")
    if (type(request["max_tokens"]) is not int or request["max_tokens"] <= 0
            or type(request["seed"]) is not int or request["stream"] is not False
            or not isinstance(request["model"], str) or not request["model"]):
        raise ValueError("PACE generation requires model, positive budget, integer seed, stream=false")
    if type(config["context_tokens"]) is not int or request["max_tokens"] > config["context_tokens"]:
        raise ValueError("PACE generation budget exceeds declared context")
    if config["stop_policy"]["mode"] != "native_terminal_only":
        raise ValueError("PACE explicit generation requires native final terminal controls")
    source = config["source"]
    if not source["path"] or len(source["sha256"]) != 64:
        raise ValueError("PACE generation requires source path and SHA256")
    if config.get("tool_parser") not in (None, "qwen3_coder"):
        raise ValueError("Unsupported explicit PACE tool parser")
    return config


def generation_config():
    path = os.environ.get(_CONFIG_ENV)
    if not path:
        return None
    digest = os.environ.get(_DIGEST_ENV)
    if digest is None:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _load(str(Path(path).resolve()), digest)


def configure_generation(path, model):
    if path is None:
        return None
    path = Path(path).expanduser().resolve(strict=True)
    os.environ[_CONFIG_ENV] = str(path)
    os.environ[_DIGEST_ENV] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = generation_config()
    if config["request"]["model"] != model:
        raise ValueError("--model differs from the explicit generation config model")
    return generation_identity()


def generation_identity():
    config = generation_config()
    if config is None:
        return None
    return {"path": str(Path(os.environ[_CONFIG_ENV]).resolve()),
            "sha256": os.environ.get(_DIGEST_ENV) or hashlib.sha256(
                Path(os.environ[_CONFIG_ENV]).read_bytes()).hexdigest(),
            "profile": config["profile"], "source": config["source"],
            "context_tokens": config["context_tokens"],
            "tool_parser": config.get("tool_parser")}


def request_configuration(kwargs, *, sdk=False):
    config = generation_config()
    if config is None:
        return dict(kwargs)
    kwargs = dict(kwargs)
    if kwargs["model"] != config["request"]["model"]:
        raise ValueError("Request model differs from the explicit PACE generation profile")
    extras = kwargs.pop("extra_body", {}) or {}
    if extras:
        raise ValueError("Explicit PACE generation cannot merge undeclared extra_body")
    for key in ("until", "stop"):
        stops = kwargs.pop(key, None) or []
        stops = [stops] if isinstance(stops, str) else stops
        unknown = set(stops) - set(config["stop_policy"]["source_stops"])
        if unknown:
            raise ValueError(f"Incompatible source stop instructions: {sorted(unknown)}")
    if kwargs.pop("n", 1) != 1:
        raise ValueError("Explicit PACE generation requires one response per request")
    kwargs.pop("store", None)
    for key in _GENERATION_FIELDS:
        kwargs.pop(key, None)
    request = config["request"]
    if sdk:
        kwargs.update({k: v for k, v in request.items() if k in _SDK_FIELDS})
        extra = {k: v for k, v in request.items() if k not in _SDK_FIELDS}
        if extra:
            kwargs["extra_body"] = extra
    else:
        kwargs.update(request)
    return kwargs


def record_response(response):
    raw = response if isinstance(response, dict) else response.model_dump(mode="json")
    retain_transport({"response": raw})
    if generation_config() is not None:
        choices = raw.get("choices")
        if not choices:
            raise GenerationFailure("Explicit PACE generation returned no choices", raw)
        for choice in choices:
            if choice.get("finish_reason") == "length":
                raise GenerationFailure("Incomplete PACE generation: finish_reason=length", raw)
            message = choice["message"]
            if message.get("incomplete_reasoning") is True:
                raise GenerationFailure("Incomplete PACE generation: incomplete_reasoning=true", raw)
            content = message.get("content")
            if (not isinstance(content, str) or not content.strip()) and not message.get("tool_calls"):
                raise GenerationFailure("Explicit PACE generation has no final content or tool calls", raw)
    return response


def record_error(exc):
    raw = {"type": type(exc).__name__, "error": str(exc)}
    response = getattr(exc, "response", None)
    if response is not None:
        raw.update(status_code=response.status_code, body=response.text)
    retain_transport(raw)
    return raw
