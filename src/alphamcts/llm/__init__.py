from .base import AlphaAgent, LLMClient, Temperatures, extract_json
from .mock import MockLLM

__all__ = ["AlphaAgent", "LLMClient", "Temperatures", "extract_json", "MockLLM", "make_client"]


def make_client(llm_cfg: dict, cwd: str | None = None) -> LLMClient:
    """Build an LLM client from the `llm` config section."""
    backend = str(llm_cfg.get("backend", "mock")).lower()
    if backend == "mock":
        return MockLLM(seed=int(llm_cfg.get("seed", 0)))
    if backend == "cursor":
        from .cursor_client import CursorLLMClient

        return CursorLLMClient(model=str(llm_cfg.get("model", "composer-2.5")), cwd=cwd)
    raise ValueError(f"Unknown llm backend: {backend!r}")
