import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np
import pytest
from models.M1.studies.m0_m1_cost_sampling_20261029.inputs import cost_weights, RGBSampler
Y = np.repeat([0, 1, 2], [1266, 6734, 796])

def test_balanced_costs_match_existing_m1_and_preserve_mean():
    for line, scrum in [(1, 1), (2, 2), (4, 2), (8, 2), (2, 1), (4, 1), (8, 1)]:
        w = cost_weights(Y, line, scrum)
        counts = np.bincount(Y)
        expected = len(Y) / (3 * counts) * [line, 1, scrum]
        expected /= np.average(expected, weights=counts)
        np.testing.assert_array_equal(w, expected)
        assert np.isclose(w[Y].mean(), 1)

def test_strided_rgb_never_reads_poisoned_discarded_vectors():
    n = 202
    values = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 2048), np.float32)
    clean = values.copy()
    keep = np.unique(np.r_[np.arange(0, n, 5), n - 1])
    values[~np.isin(np.arange(n), keep)] = np.nan
    sampler = RGBSampler(values, 5)
    x, m = sampler.window(11, 175)
    np.testing.assert_allclose(x[:175], clean[11:186])
    assert np.isfinite(x).all()
    assert m.sum() == 175 and (not m[175:].any()) and (not x[175:].any())
    assert sampler.reader.seen <= set(keep.tolist())

def test_native_control_and_final_endpoint():
    values = np.random.default_rng(42).normal(size=(201, 2048)).astype(np.float32)
    s = RGBSampler(values, 1)
    x, m = s.window(26, 175)
    np.testing.assert_array_equal(x[:175], values[26:201])
    sparse = RGBSampler(values, 25)
    assert sparse.indices[-1] == 200
    x, m = sparse.window(196, 5)
    np.testing.assert_array_equal(x[4], values[-1])
    assert m.sum() == 5 and x[5:].sum() == 0

@pytest.mark.parametrize('labels,line,scrum', [([], 2, 1), ([0, 1], 2, 1), ([0, 1, 2], float('nan'), 1), ([0, 1, 2], 2, 0)])
def test_invalid_costs_and_missing_classes(labels, line, scrum):
    with pytest.raises(ValueError):
        cost_weights(labels, line, scrum)
