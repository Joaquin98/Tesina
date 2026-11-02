import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json, hashlib, zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__).parents[0]
OLD = ROOT / 'data/m0_hard_line_cost4_d2_vf_20261027'
BASE = ROOT / 'data/central_recall_cost_d2_vf_20261027'
OOF = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
DATA = ROOT / 'data/multicrop_d2_stride25_vf_20261026'
sys.path[:0] = [str(source_directory(HERE)), str(source_directory(BASE))]
from hard import sample_weights
from evaluate import bootstrap

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def main():
    completion = json.loads(_resolve_model_artifact(HERE / 'completion.json').read_text())
    assert completion['complete'] and completion['all_weights_saved_before_vf']
    proto = json.loads(_resolve_model_artifact(HERE / 'protocol.json').read_text())
    assert proto['hard_multiplier'] == 3
    for p, h in proto['hashes'].items():
        assert sha(Path(p)) == h, p
    rows = json.loads(_resolve_model_artifact(DATA / 'manifest.json').read_text())
    train = [r for r in rows if r['domain'] != 'VF']
    frame = pd.DataFrame([r for r in rows if r['domain'] == 'VF'])
    labels = ['line', 'other', 'scrum']
    y = np.array([labels.index(r['tag']) for r in train])
    truth = np.array([labels.index(r['tag']) for r in frame.to_dict('records')])
    results = []
    preds = {'central': [], 'x2': [], 'x3': []}
    source = pd.read_csv(_resolve_model_artifact(HERE / 'results.csv'))
    for seed in [17, 42, 2026]:
        out = HERE / f'seed_{seed}'
        oof = _load_model_result_array(OOF / f'seed_{seed}/oof.npy')[:, 0].argmax(1)
        w, hard = sample_weights(y, oof)
        np.testing.assert_array_equal(hard, _load_model_result_array(out / 'hard_mask.npy'))
        np.testing.assert_allclose(w, _load_model_result_array(out / 'sample_weights.npy'))
        info = json.loads(_resolve_model_artifact(out / 'selected.json').read_text())
        assert sha(out / 'model.weights.h5') == info['checkpoint_sha256'] and len(pd.read_csv(_resolve_model_artifact(out / 'history.csv'))) == 15
        for name, path in [('central', BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), ('x2', OLD / f'seed_{seed}/vf_probabilities.npy'), ('x3', out / 'vf_probabilities.npy')]:
            p = _load_model_result_array(path)
            assert p.shape == (688, 3)
            pred = p.argmax(1)
            preds[name].append(pred)
            pr, re, f, sup = precision_recall_fscore_support(truth, pred, labels=[0, 1, 2], zero_division=0)
            d = dict(seed=seed, variant=name, accuracy=float(accuracy_score(truth, pred)), f1_macro=float(f.mean()))
            for j, k in enumerate(labels):
                d.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(((truth == j) & (pred == j)).sum()), k + '_fn': int(((truth == j) & (pred != j)).sum()), k + '_fp': int(((truth != j) & (pred == j)).sum())})
            if name != 'x2':
                saved = source[(source.seed == seed) & (source.variant == ('central' if name == 'central' else 'hard_line'))].iloc[0]
                for key, val in d.items():
                    if key not in ['seed', 'variant']:
                        np.testing.assert_allclose(saved[key], val, atol=1e-12)
            results.append(d)
    df = pd.DataFrame(results)
    df.to_csv(HERE / 'combined_results.csv', index=False)
    ci = bootstrap(truth, preds['x2'], preds['x3'], frame.section, frame.start_s)
    (HERE / 'x3_vs_x2_bootstrap.json').write_text(json.dumps(ci, indent=2))
    backup = ROOT / 'model_backups/m0_hard_line3_cost4_d2_vf_20261027.zip'
    files = [p for p in _model_artifact_glob(HERE, '*', recursive=True) if _resolve_model_artifact(p).is_file() and p.suffix in ['.h5', '.npy', '.py', '.json', '.csv'] and (p.name not in ['status.json', 'error.json'])]
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(HERE))
    with zipfile.ZipFile(backup) as z:
        assert z.testzip() is None
        for p in files:
            assert hashlib.sha256(z.read(p.relative_to(HERE).as_posix())).hexdigest() == sha(p)
    result = dict(passed=True, means=df.groupby('variant').mean(numeric_only=True).to_dict(orient='index'), x3_vs_x2=ci, backup=str(backup), backup_sha256=sha(backup), models=completion['models'])
    (HERE / 'analysis/verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
if __name__ == '__main__':
    main()
