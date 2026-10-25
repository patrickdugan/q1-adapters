"""Offline contract tests for the Hyperspace pod gateway adapter."""

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hyperspace_pod import (  # noqa: E402
    ensure_resume_identity, pod_key, preflight_models, request_options, validate_base_url,
)


class HyperspacePodTests(unittest.TestCase):
    def test_loopback_and_remote_tls(self):
        self.assertEqual(validate_base_url("http://127.0.0.1:8080/v1/"),
                         "http://127.0.0.1:8080/v1")
        self.assertEqual(validate_base_url("https://pod.example/v1"), "https://pod.example/v1")
        for url in ("http://pod.example/v1", "http://user:pass@localhost/v1",
                    "https://pod.example/v1?token=secret", "https://pod.example/api"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_base_url(url)

    def test_key_comes_only_from_environment(self):
        with patch.dict(os.environ, {"HYPERSPACE_POD_API_KEY": "test-key"}):
            self.assertEqual(pod_key("HYPERSPACE_POD_API_KEY"), "test-key")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                pod_key("HYPERSPACE_POD_API_KEY")

    def test_preflight_exact_model_and_auth_header(self):
        payload = b'{"data":[{"id":"qwen3.6-27b"},{"id":"other"}]}'
        with patch("hyperspace_pod.urllib.request.urlopen", return_value=io.BytesIO(payload)) as mocked:
            ids = preflight_models("http://localhost:8080/v1", "test-key", "qwen3.6-27b")
            self.assertIn("qwen3.6-27b", ids)
            request = mocked.call_args.args[0]
            self.assertEqual(request.full_url, "http://localhost:8080/v1/models")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        with patch("hyperspace_pod.urllib.request.urlopen", return_value=io.BytesIO(payload)):
            with self.assertRaises(RuntimeError):
                preflight_models("http://localhost:8080/v1", "test-key", "qwen3.6-32b")

    def test_pod_requests_exclude_llama_specific_knobs(self):
        pod = request_options("hyperspace", 123, 20)
        self.assertEqual(pod, {"seed": 123})
        llama = request_options("llama", 123, 20)
        self.assertFalse(llama["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(llama["extra_body"]["top_k"], 20)

    def test_resume_requires_matching_model_and_route(self):
        expected = {"backend": "hyperspace", "base_url": "http://localhost:8080/v1",
                    "model": "qwen3.6-27b"}
        ensure_resume_identity({"endpoint": expected}, expected)
        with self.assertRaises(RuntimeError):
            ensure_resume_identity({"endpoint": {**expected, "model": "other"}}, expected)
        with self.assertRaises(RuntimeError):
            ensure_resume_identity({"id": "old-run"}, expected)


if __name__ == "__main__":
    unittest.main()
