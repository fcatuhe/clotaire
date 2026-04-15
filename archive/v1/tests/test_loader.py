"""Tests for model loader."""

from __future__ import annotations


from kloter.models.loader import clear_cache, _cache


class TestLoaderCache:
    """Tests for model caching behavior."""

    def test_clear_cache(self):
        """clear_cache empties the model cache."""
        _cache["test_key"] = "test_value"
        assert len(_cache) > 0
        clear_cache()
        assert len(_cache) == 0

    def test_cache_is_dict(self):
        """Cache is a plain dict."""
        clear_cache()
        assert isinstance(_cache, dict)
