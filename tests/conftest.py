import pytest

from backend import llm_client


@pytest.fixture(autouse=True)
def _isolate_llm_disk_cache(tmp_path, monkeypatch):
    """
    Every test gets its own throwaway cache directory, so tests that don't
    care about caching (most of them) never read or write the real
    data/cache/ - and never silently pollute each other via a shared key.
    Tests that specifically exercise caching still patch CACHE_DIR
    themselves with their own tmp_path, which simply overrides this for
    their duration.
    """
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path / "llm_cache")
