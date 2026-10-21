import numpy as np
import pytest

from vector_portability.core import Dataset, Filter, exact, generate, metadata, metrics, vectors


def test_predicate_conjunction():
    meta = np.array([[1, 1, 95], [1, 1, 94], [0, 1, 99], [1, 0, 99]])
    where = Filter(1, 1, 95)
    assert where.mask(meta).tolist() == [True, False, False, False]
    assert where.sql() == (" AND tenant = ? AND active = ? AND bucket >= ?", [1, 1, 95])


def test_exact_ties_and_metrics():
    x = np.array([[1, 0], [-1, 0], [0, 2]], dtype=np.float32)
    meta = np.zeros((3, 3), dtype=np.int64)
    assert exact(x, meta, np.zeros(2, dtype=np.float32), 2, Filter()) == [0, 1]
    assert metrics([0], [0, 1], 100, 2) == {"recall": 0.5, "result_loss": 0.5}
    assert metrics([], [], 0, 2) == {"recall": None, "result_loss": None}
    with pytest.raises(ValueError):
        metrics([0, 0], [0], 2, 2)


def test_dataset_roundtrip_and_validation(tmp_path):
    data = generate(20, 4, 3)
    path = tmp_path / "data.npz"
    data.save(path)
    restored = Dataset.load(path)
    np.testing.assert_array_equal(restored.base, data.base)
    np.testing.assert_array_equal(restored.state(True)[0][data.update_ids], data.update_vectors)
    with pytest.raises(ValueError):
        data.save(path)
    data.update_ids[0] = 999
    with pytest.raises(ValueError):
        data.__post_init__()


@pytest.mark.parametrize("value", [Filter, 1.2, True, "0"])
def test_bad_filters(value):
    with pytest.raises(ValueError):
        Filter(tenant=value)


def test_bad_arrays():
    with pytest.raises(ValueError):
        vectors([[float("nan")]])
    with pytest.raises(ValueError):
        metadata([[1, 2, 3]], 1)
    with pytest.raises(ValueError):
        metadata([[1.0, 1.0, 3.0]], 1)
    with pytest.raises(ValueError):
        generate(1)
    with pytest.raises(ValueError):
        Filter(active=2)
