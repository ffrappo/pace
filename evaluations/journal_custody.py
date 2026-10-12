"""Synchronous custody of exact PACE journal bytes in one private snapshot object."""
import hashlib
import json
import os
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class CustodyError(RuntimeError):
    """Off-worker custody failed, independently of any committed model observation."""

    journal = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class JournalCustody:
    SCHEMA = "pace-score-journal-v1"

    @classmethod
    def from_env(cls):
        path = os.environ.get("PACE_JOURNAL_CUSTODY")
        return cls(path) if path is not None else None

    def __init__(self, config_path):
        try:
            config = json.loads(Path(config_path).read_bytes())
        except (OSError, ValueError) as exc:
            raise CustodyError(f"PACE custody config unreadable ({type(exc).__name__})") from None
        if not isinstance(config, dict) or set(config) != {"put_url", "get_url", "content_type"}:
            raise CustodyError("PACE custody config requires put_url, get_url and content_type")
        if config["content_type"] != "application/json":
            raise CustodyError("PACE custody content_type must be application/json")
        try:
            urls = [urlsplit(config[key]) for key in ("put_url", "get_url")]
            if any(url.scheme != "https" or not url.hostname or not url.path or not url.query
                   or url.fragment or url.username or url.password for url in urls):
                raise ValueError
            if (urls[0].netloc, urls[0].path) != (urls[1].netloc, urls[1].path):
                raise ValueError
        except (TypeError, ValueError, AttributeError):
            raise CustodyError("PACE custody URLs must bind one exact HTTPS object") from None
        self._put_url, self._get_url = config["put_url"], config["get_url"]
        self._content_type = config["content_type"]
        self._http = build_opener(_NoRedirect())

    def _request(self, method, data=None):
        url = self._put_url if method == "PUT" else self._get_url
        headers = {"Content-Type": self._content_type} if method == "PUT" else {}
        try:
            request = Request(url, data=data, headers=headers, method=method)
            with self._http.open(request, timeout=120) as response:
                if not 200 <= response.status < 300:
                    raise CustodyError(f"PACE custody {method} HTTP {response.status}")
                return response.read() if method == "GET" else b""
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise CustodyError(f"PACE custody {method} HTTP {status}") from None
        except (URLError, OSError, ValueError, HTTPException) as exc:
            # Exception text and response bodies can contain signed URLs. Never expose either.
            raise CustodyError(f"PACE custody {method} transport failure ({type(exc).__name__})") from None

    def publish(self, binding_bytes, observations_bytes):
        files = {}
        for name, data in (("binding.json", binding_bytes), ("observations.jsonl", observations_bytes)):
            files[name] = {"text": data.decode("utf-8"), "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}
        payload = (json.dumps({"schema": self.SCHEMA, "files": files}, ensure_ascii=False,
                              allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        self._request("PUT", payload)
        actual = self._request("GET")
        if actual != payload or hashlib.sha256(actual).digest() != hashlib.sha256(payload).digest():
            raise CustodyError("PACE custody GET snapshot byte/digest mismatch")
        snapshot = json.loads(actual)
        if snapshot["schema"] != self.SCHEMA:
            raise CustodyError("PACE custody GET schema mismatch")
        for name, expected in files.items():
            retained = snapshot["files"][name]
            data = retained["text"].encode("utf-8")
            if (len(data) != retained["bytes"] or hashlib.sha256(data).hexdigest() != retained["sha256"]
                    or retained != expected):
                raise CustodyError("PACE custody GET journal digest mismatch")
