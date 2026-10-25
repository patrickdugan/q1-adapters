"""OpenAI-compatible Hyperspace Pod adapter for the out-of-process Q1 evaluator.

The gateway is treated as a remote model provider even when reached via localhost.
No Hyperspace CLI binary or network dependency is imported by this module.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def validate_base_url(url: str) -> str:
    """Accept /v1 gateways; require HTTPS when the gateway is not loopback."""
    url = url.rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.path.endswith("/v1")):
        raise ValueError("gateway URL must end in /v1 and contain no credentials or query")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("remote pod gateways must use HTTPS")
    return url


def pod_key(env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise ValueError(f"set {env_name} to a pod-scoped key before running the evaluator")
    return key


def preflight_models(base_url: str, key: str, model: str, timeout: float = 15.0) -> list[str]:
    """Fail before a paid/model-changing sweep if the pod omits the requested ID."""
    url = validate_base_url(base_url) + "/models"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"pod model preflight returned HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError("pod model preflight could not reach gateway") from None
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise RuntimeError("pod gateway returned an invalid /v1/models response")
    ids = sorted(x["id"] for x in data["data"] if isinstance(x, dict) and isinstance(x.get("id"), str))
    if model not in ids:
        raise RuntimeError(f"model {model!r} missing from pod /v1/models; available: {ids[:12]}")
    return ids


def request_options(backend: str, seed: int, top_k: int) -> dict:
    """Keep llama.cpp extensions off the portable pod gateway request."""
    if backend == "hyperspace":
        # Seed is a standard optional chat field; gateway/model support varies.
        return {"seed": seed}
    if backend == "llama":
        return {"extra_body": {
            "top_k": top_k, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }}
    raise ValueError(f"unsupported inference backend: {backend}")


def ensure_resume_identity(row: dict, expected: dict) -> None:
    """Guard against appending pod samples to another model or inference route."""
    if not isinstance(row, dict) or row.get("endpoint") != expected:
        raise RuntimeError(
            "existing output contains another endpoint or an older untagged run; "
            "start with a distinct --out"
        )
