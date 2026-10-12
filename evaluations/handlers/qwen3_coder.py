"""Strict non-streaming Qwen3 Coder XML adapter for the actual BFCL owner.

Grammar and schema conversions: vLLM v0.19.0 qwen3coder_tool_parser.py.
Unlike its recovery paths, malformed tags/types fail rather than invent a value.
"""
import hashlib
import json
import math
import re
from copy import deepcopy

from evaluations.generation import GenerationFailure

_TOOL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"\s*<function=([^<>]+)>(.*?)</function>\s*", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^<>]+)>(.*?)</parameter>", re.DOTALL)


def _value(text, schema):
    # The published grammar removes one leading and one trailing newline.
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    if text.lower() == "null":
        return None
    kind = schema.get("type")
    if kind is None and "anyOf" in schema:
        return json.loads(text)
    if not isinstance(kind, str):
        raise ValueError("Qwen tool parameter needs an explicit schema type")
    kind = kind.strip().lower()
    if kind in {"string", "str", "text", "varchar", "char", "enum"}:
        return text
    if kind.startswith(("int", "uint", "long", "short", "unsigned")):
        return int(text)
    if kind.startswith(("num", "float")):
        number = float(text)
        if not math.isfinite(number):
            raise ValueError("Nonfinite Qwen tool parameter")
        return number if number != int(number) else int(number)
    if kind in {"boolean", "bool", "binary"}:
        if text.lower() not in {"true", "false"}:
            raise ValueError("Qwen boolean parameter must be true or false")
        return text.lower() == "true"
    if kind in {"object", "array", "arr"} or kind.startswith(("dict", "list")):
        value = json.loads(text)
        expected = dict if kind == "object" or kind.startswith("dict") else list
        if not isinstance(value, expected):
            raise ValueError(f"Qwen parameter does not match {kind}")
        return value
    raise ValueError(f"Unsupported Qwen parameter schema type: {kind}")


def adapt_response(response, tools):
    """Return an OpenAI-shaped copy; never overwrite the retained native response."""
    from openai.types.chat import ChatCompletion
    raw = response if isinstance(response, dict) else response.model_dump(mode="json")
    converted = deepcopy(raw)
    try:
        for choice in converted["choices"]:
            message = choice["message"]
            content = message.get("content") or ""
            markers = any(tag in content for tag in (
                "<tool_call>", "<function=", "<parameter=", "</tool_call>", "</function>", "</parameter>"))
            if not markers:
                continue
            if message.get("tool_calls"):
                raise ValueError("Native XML and structured tool calls both supplied")
            by_name = {}
            for tool in tools:
                function = tool["function"]
                if function["name"] in by_name:
                    raise ValueError("Duplicate tool schema name")
                by_name[function["name"]] = function["parameters"]["properties"]
            matches = list(_TOOL.finditer(content))
            if not matches:
                raise ValueError("Unclosed or unwrapped Qwen tool call")
            calls = []
            cursor = matches[0].start()
            prefix = content[:cursor]
            if any(tag in prefix for tag in ("<function=", "<parameter=", "</tool_call>")):
                raise ValueError("Malformed content before Qwen tool call")
            for match in matches:
                if content[cursor:match.start()].strip():
                    raise ValueError("Unexpected text between Qwen tool calls")
                function = _FUNCTION.fullmatch(match.group(1))
                if function is None:
                    raise ValueError("Malformed Qwen function declaration")
                name, body = function.groups()
                if name not in by_name:
                    raise ValueError(f"Qwen function absent from actual request tools: {name}")
                arguments, end = {}, 0
                for parameter in _PARAMETER.finditer(body):
                    if body[end:parameter.start()].strip():
                        raise ValueError("Malformed Qwen parameter declaration")
                    key, text = parameter.groups()
                    if key in arguments or key not in by_name[name]:
                        raise ValueError(f"Duplicate or undeclared Qwen parameter: {key}")
                    arguments[key] = _value(text, by_name[name][key])
                    end = parameter.end()
                if body[end:].strip():
                    raise ValueError("Unclosed Qwen parameter declaration")
                identifier = hashlib.sha256(f"{raw['id']}:{choice['index']}:{len(calls)}".encode()).hexdigest()[:24]
                calls.append({"id": "call_" + identifier, "type": "function",
                              "function": {"name": name, "arguments": json.dumps(
                                  arguments, ensure_ascii=False, allow_nan=False)}})
                cursor = match.end()
            if content[cursor:].strip():
                raise ValueError("Unexpected content after Qwen tool calls")
            message["content"] = prefix or None
            message["tool_calls"] = calls
        return ChatCompletion.model_validate(converted)
    except (ValueError, KeyError, TypeError) as exc:
        raise GenerationFailure(f"Qwen3 Coder tool parse failed: {exc}", raw) from exc
