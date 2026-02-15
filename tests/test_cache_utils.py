"""Tests for mlx_qwen3_asr/cache_utils.py."""

from mlx_qwen3_asr.cache_utils import LRUCache


def test_lru_cache_eviction_order():
    cache = LRUCache[str, int](max_entries=2)
    assert cache.put("a", 1) is None
    assert cache.put("b", 2) is None

    # Touch "a" so "b" becomes least-recent.
    assert cache.get("a") == 1
    evicted = cache.put("c", 3)

    assert evicted == ("b", 2)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_lru_cache_capacity_change_trims_oldest():
    cache = LRUCache[str, int](max_entries=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.set_max_entries(1)

    assert len(cache) == 1
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.get("c") == 3
