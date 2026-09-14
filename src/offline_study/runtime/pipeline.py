"""Bounded pipeline helpers for overlapping decode and artifact serialization."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import torch


T = TypeVar("T")
_ITEM = object()
_DONE = object()
_ERROR = object()


class PrefetchIterator(Iterator[T]):
    """Consume one iterator on a bounded producer thread.

    The source iterator has exactly one owner. This overlaps storage/decode with GPU
    work without concurrently touching a dataset object from multiple workers.
    """

    def __init__(self, iterable: Iterable[T], depth: int):
        if depth < 0:
            raise ValueError("Prefetch depth cannot be negative")
        self._iterator = iter(iterable)
        self._depth = depth
        self._queue: queue.Queue = queue.Queue(maxsize=max(depth, 1))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.producer_seconds = 0.0
        self.consumer_wait_seconds = 0.0
        self.items_produced = 0
        if depth:
            self._thread = threading.Thread(target=self._produce, name="jepa-batch-prefetch", daemon=True)
            self._thread.start()

    def __iter__(self) -> PrefetchIterator[T]:
        return self

    def _put(self, value) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                try:
                    value = next(self._iterator)
                except StopIteration:
                    self.producer_seconds += time.perf_counter() - started
                    self._put((_DONE, None))
                    return
                self.producer_seconds += time.perf_counter() - started
                self.items_produced += 1
                self._put((_ITEM, value))
        except Exception as exc:
            self._put((_ERROR, exc))

    def __next__(self) -> T:
        if not self._depth:
            started = time.perf_counter()
            try:
                value = next(self._iterator)
            except StopIteration:
                self.producer_seconds += time.perf_counter() - started
                raise
            elapsed = time.perf_counter() - started
            self.producer_seconds += elapsed
            self.consumer_wait_seconds += elapsed
            self.items_produced += 1
            return value
        started = time.perf_counter()
        kind, value = self._queue.get()
        self.consumer_wait_seconds += time.perf_counter() - started
        if kind is _ITEM:
            return value
        self.close()
        if kind is _ERROR:
            raise value
        raise StopIteration

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)


def _save_tensor_bundle(path: Path, payload: object) -> float:
    started = time.perf_counter()
    torch.save(payload, path)
    return time.perf_counter() - started


class AsyncCacheWriter:
    """Serialize CPU tensor bundles on one bounded background worker."""

    def __init__(self, depth: int):
        if depth < 1:
            raise ValueError("Cache queue depth must be positive")
        self._depth = depth
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jepa-cache")
        self._pending: deque[Future] = deque()
        self.writer_seconds = 0.0
        self.consumer_wait_seconds = 0.0

    def _collect_one(self) -> None:
        started = time.perf_counter()
        self.writer_seconds += self._pending.popleft().result()
        self.consumer_wait_seconds += time.perf_counter() - started

    def submit(self, path: Path, payload: object) -> None:
        if len(self._pending) >= self._depth:
            self._collect_one()
        self._pending.append(self._executor.submit(_save_tensor_bundle, path, payload))

    def close(self) -> None:
        try:
            while self._pending:
                self._collect_one()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
