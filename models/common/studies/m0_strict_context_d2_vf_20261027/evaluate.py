import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from rule import decide
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
BASE = ROOT / 'data/central_recall_cost_d2_vf_20261027'
labels = ['line', 'other', 'scrum']

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def main():
    file = HERE / 'thresholds.json'
    frozen = sha(file)
    spec = json.loads(_resolve_model_artifact(file).read_text())
    assert spec['frozen']
    for name, expected in spec['hashes'].items():
        assert sha(Path(name)) == expected, name
    frame = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    assert len(frame) == 688
    y = np.array([labels.index(t) for t in frame.tag])
    assert np.bincount(y).tolist() == [68, 581, 39]
    records = []
    cases = []
    for seed in [17, 42, 2026]:
        thresholds = spec['seeds'][str(seed)]['thresholds']
        p = _load_model_result_array(P / f'seed_{seed}/vf_views.npy')
        np.testing.assert_allclose(p[:, 0], _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        central = p[:, 0].argmax(1)
        pred = decide(p, thresholds)
        for variant, out in [('central', central), ('strict', pred)]:
            pr, re, f, _ = precision_recall_fscore_support(y, out, labels=[0, 1, 2], zero_division=0)
            cm = confusion_matrix(y, out, labels=[0, 1, 2])
            row = dict(seed=seed, variant=variant, accuracy=float(accuracy_score(y, out)), f1_macro=float(f.mean()), changes=int((out != central).sum()))
            for j, k in enumerate(labels):
                row.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(cm[j, j]), k + '_fn': int(cm[j].sum() - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
            records.append(row)
        cases.extend((dict(seed=seed, clip_id=frame.clip_id.iloc[i], human=labels[y[i]], central=labels[central[i]], strict=labels[pred[i]], changed=bool(pred[i] != central[i])) for i in range(len(y))))
        if thresholds == [None, None]:
            assert np.array_equal(central, pred)
    assert sha(file) == frozen
    pd.DataFrame(records).to_csv(HERE / 'results.csv', index=False)
    pd.DataFrame(cases).to_csv(HERE / 'predictions.csv', index=False)
    info = dict(complete=True, threshold_file_sha256=frozen, thresholds_fixed_using_D2_before_VF=True, means=pd.DataFrame(records).groupby('variant').mean(numeric_only=True).to_dict(orient='index'), probability_threshold=True, unchanged_when_disabled=True, limitations=['90% selection target was chosen as a strict criterion, not a guarantee', 'no qualifying correction candidate on D2; overrides disabled', 'confidence is not calibrated probability of correctness', 'VF previously observed, no independent external evaluation', 'OOF selection models trained on 2/3 D2; final VF models trained on all D2', 'three seeds not an ensemble'])
    (HERE / 'results.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info))
if __name__ == '__main__':
    main()
