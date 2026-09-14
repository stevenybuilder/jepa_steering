import importlib.util
import io
from pathlib import Path
import time

import pytest

spec = importlib.util.spec_from_file_location('parallel_download',
    Path(__file__).resolve().parents[3] / 'scripts/vast/fresh_parallel_download.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('connections', [4, 16, 32])
def test_ranges_cover_exactly_once(connections):
    spans = m.segments(m.SIZE, connections)
    assert spans[0][0] == 0 and spans[-1][1] == m.SIZE - 1
    assert all(a[1] + 1 == b[0] for a, b in zip(spans, spans[1:]))
    assert sum(b - a + 1 for a, b in spans) == m.SIZE


class Response(io.BytesIO):
    status = 206
    headers = {'Content-Range': 'bytes 2-5/8'}


def test_range_preserves_offsets_and_checks_server():
    calls = []
    def opener(request, timeout):
        assert request.headers['Range'] == 'bytes=2-5'
        assert timeout == 15
        return Response(b'abcd')
    assert m.fetch_range({'url': 'https://example.test', 'token': 'dummy', 'size': 8},
        2, 5, time.monotonic() + 2, lambda *args: calls.append(args), opener) == 4
    assert calls == [(2, b'abcd')]


@pytest.mark.parametrize('status,header,data', [(200, 'bytes 2-5/8', b'abcd'),
    (206, 'bytes 2-5/9', b'abcd'), (206, 'bytes 2-5/8', b'ab')])
def test_bad_range_or_truncation_rejected(status, header, data):
    def opener(*args, **kwargs):
        result = Response(data)
        result.status = status
        result.headers = {'Content-Range': header}
        return result
    with pytest.raises(ValueError):
        m.fetch_range({'url': 'https://example.test', 'token': 'dummy', 'size': 8},
            2, 5, time.monotonic() + 2, lambda *args: None, opener)


def test_expired_deadline_rejected():
    with pytest.raises(TimeoutError):
        m.fetch_range({'url': 'https://example.test', 'token': 'dummy', 'size': 8},
            2, 5, time.monotonic() - 1, lambda *args: None, lambda *a, **k: Response(b'abcd'))
