"""LLM backend on Cursor's Composer model via the Cursor SDK (`cursor-sdk`).

Each `complete` call is a one-shot `Agent.prompt` with the configured model
(default: composer-2.5) on the local runtime. Requires `CURSOR_API_KEY`.

Note: the Cursor SDK does not expose a sampling temperature, so the `temperature`
argument is accepted for interface compatibility and ignored.

Calls are wrapped with a hard timeout and retries so a hung SDK call cannot stall a
multi-hour mining run.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os

logger = logging.getLogger("alphamcts")


class CursorLLMClient:
    def __init__(
        self,
        model: str = "composer-2.5",
        api_key: str | None = None,
        cwd: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("CURSOR_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is not set. Export it or add it to .env "
                "(see .env.example) to use the Cursor Composer backend."
            )
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout
        self.max_retries = max_retries

    def _call_once(self, prompt: str) -> str:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

        result = Agent.prompt(
            "You are an expert quantitative researcher. Respond with the requested output "
            "only; when JSON is requested, output a single JSON object.\n\n" + prompt,
            AgentOptions(
                api_key=self.api_key,
                model=self.model,
                local=LocalAgentOptions(cwd=self.cwd),
            ),
        )
        if result.status != "finished":
            raise RuntimeError(f"Cursor agent run ended with status {result.status!r} (id={result.id})")
        return result.result or ""

    def complete(self, prompt: str, temperature: float = 1.0) -> str:  # noqa: ARG002
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self._call_once, prompt)
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                last_error = RuntimeError(f"Cursor SDK call timed out after {self.timeout:.0f}s")
                logger.warning("LLM call timeout (attempt %d/%d)", attempt + 1, self.max_retries + 1)
            except Exception as exc:  # noqa: BLE001 - retry transient SDK/network errors
                last_error = exc
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, exc)
            finally:
                executor.shutdown(wait=False)
        assert last_error is not None
        raise last_error
