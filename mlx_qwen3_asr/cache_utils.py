"""Small bounded LRU cache utilities used by process-local holders."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Bounded in-memory LRU cache."""

    def __init__(self, max_entries: int):
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._max_entries = int(max_entries)
        self._data: OrderedDict[K, V] = OrderedDict()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def set_max_entries(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._max_entries = int(max_entries)
        self._trim_to_capacity()

    def get(self, key: K) -> V | None:
        value = self._data.get(key)
        if value is None:
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> tuple[K, V] | None:
        if key in self._data:
            self._data[key] = value
            self._data.move_to_end(key)
            return None

        self._data[key] = value
        if len(self._data) <= self._max_entries:
            return None
        return self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def _trim_to_capacity(self) -> None:
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)
