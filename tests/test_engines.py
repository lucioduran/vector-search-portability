from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vector_portability.core import FILTERS, Filter, exact, generate
from vector_portability.engines import create


@pytest.fixture(params=["hnswlib", "sqlite-vec"])
def engine(request, tmp_path):
    value = create(request.param, tmp_path / request.param, 4)
    yield value
    value.close()


def test_real_engine_contract_and_updates(engine):
    data = generate(80, 4, 5)
    engine.build(data.base, data.meta)
    for updated in (False, True):
        if updated:
            engine.update(data.update_ids, data.update_vectors, data.update_meta)
        x, meta = data.state(updated)
        for where in FILTERS.values():
            for q in data.queries:
                ids, distances = engine.search(q, 10, where)
                assert ids == exact(x, meta, q, 10, where)
                expected = np.sum((x[ids].astype(float) - q) ** 2, axis=1)
                np.testing.assert_allclose(distances, expected, rtol=1e-5)
    assert engine.snapshot_bytes() > 0
    with ThreadPoolExecutor(4) as pool:
        answers = list(pool.map(lambda _: engine.search(data.queries[0], 10, Filter()), range(12)))
    assert all(answer == answers[0] for answer in answers)


def test_invalid_lifecycle(engine):
    data = generate(20, 4, 1)
    with pytest.raises(ValueError, match="not ready"):
        engine.search(data.queries[0], 2, Filter())
    with pytest.raises(ValueError, match="dimension"):
        engine.build(np.ones((2, 3), dtype=np.float32), data.meta[:2])
    engine.build(data.base, data.meta)
    with pytest.raises(ValueError, match="already"):
        engine.build(data.base, data.meta)
    for q, k in [(np.ones(3), 2), (np.full(4, np.nan), 2), (np.ones(4), 0)]:
        with pytest.raises(ValueError):
            engine.search(q, k, Filter())
    for ids in [np.array([-1]), np.array([20]), np.array([0, 0])]:
        with pytest.raises(ValueError):
            engine.update(ids, data.base[: len(ids)], data.meta[: len(ids)])


def test_underfull_and_metadata_change(engine):
    x = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.float32)
    meta = np.array([[0, 1, 95], [1, 1, 96], [1, 0, 10]])
    engine.build(x, meta)
    assert engine.search(x[0], 10, Filter(tenant=0))[0] == [0]
    engine.update(np.array([0]), x[2:3], np.array([[1, 0, 1]]))
    assert engine.search(x[0], 10, Filter(tenant=0))[0] == []
    assert engine.search(x[0], 10, Filter(active=1))[0] == [1]
