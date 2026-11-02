"""Validate actual reference geometry, head, masks, serialization and fold splits.
This engineering test produces no test-set accuracy scores.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json, sys
from pathlib import Path
import numpy as np
import keras
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.experiment import HERE, ROOT, OUT, new_head, weights_digest, write_json
from models.common.comparison.protocol import PV_FOLDS, RunSpec
from models.common.comparison.data import build_master_manifest, split_for_run
from models.common.comparison.models import build_dual_model
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.preflight import halfscale, center_crop
rng = np.random.default_rng(20261029)
frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
half = halfscale(frame)
assert half.shape == (360, 640, 3)
a, b = (center_crop(half, 224), center_crop(half, 320))
assert a.shape == (224, 224, 3) and b.shape == (320, 320, 3)
assert np.array_equal(a, center_crop(b, 224))
try:
    halfscale(np.zeros((480, 1280, 3), np.uint8))
except ValueError:
    pass
else:
    raise AssertionError('An incompatible source was silently padded, enlarged or accepted')
m = build_master_manifest(write=False)
splits = []
for f in PV_FOLDS:
    spec = RunSpec('PV', 'M0', 'H2', 'D2', f.fold, 42, f.train_groups, f.development_group, f.test_group)
    tr, dv, te = split_for_run(m, spec)
    assert set(tr.loc[tr.domain == 'CV', 'group_id']) == set(f.train_groups)
    assert set(dv.group_id) == {f.development_group} and set(te.group_id) == {f.test_group}
    assert len(tr[tr.domain == 'CH']) == 4082
    splits.append(dict(fold=f.fold, n_train=len(tr), n_dev=len(dv), n_test=len(te)))
reference = keras.models.load_model(_resolve_model_artifact(ROOT / 'models/M0/checkpoints/best_model/best_model.keras'), compile=False)
head = new_head(17)
assert head.count_params() == reference.count_params()
for name in ('bidirectional', 'bidirectional_1', 'dense', 'dense_1'):
    assert head.get_layer(name).get_config()['trainable'] == reference.get_layer(name).get_config()['trainable']
    assert [w.shape for w in head.get_layer(name).get_weights()] == [w.shape for w in reference.get_layer(name).get_weights()]
assert head.get_layer('bidirectional').forward_layer.return_sequences
assert not head.get_layer('bidirectional_1').forward_layer.return_sequences
assert weights_digest(head) == weights_digest(new_head(17))
folder = OUT / 'head_contract'
folder.mkdir(exist_ok=True)
head = new_head(17)
ih = weights_digest(head)
head.save(folder / 'initial.keras')
loaded = keras.models.load_model(_resolve_model_artifact(folder / 'initial.keras'), compile=False)
assert weights_digest(loaded) == ih
for size in (224, 320):
    with _load_model_result_array(OUT / 'smoke' / f'features_{size}.npz') as p:
        values = p['features']
        x = np.zeros((2, 200, 2048), np.float32)
        x[0, :8] = values[:8]
        x[1, :8] = values[8:]
    mask = np.zeros((2, 200), bool)
    mask[:, :8] = True
    model = keras.models.load_model(_resolve_model_artifact(folder / 'initial.keras'), compile=False)
    model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy')
    predicted = np.asarray(model.predict_on_batch(dict(features=x, mask=mask)))
    assert predicted.shape == (2, 3) and np.isfinite(predicted).all()
    loss = float(model.train_on_batch(dict(features=x, mask=mask), np.array([0, 1])))
    assert np.isfinite(loss) and weights_digest(model) != ih
result = dict(status='passed', reference_params=reference.count_params(), new_params=head.count_params(), initial_weights_sha256=ih, paired_seeds=[17, 42, 2026], folds=splits, note='Real source features and masked training update tested at both crop sizes. Engineering validation only, not a reduced scientific experiment.')
write_json(folder / 'validation.json', result)
print(json.dumps(result, indent=2), flush=True)
