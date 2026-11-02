"""Independent saved-prediction verification, report checks and recoverable backup."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import sys, time, hashlib, zipfile
import pandas as pd
from urllib.request import urlopen, Request
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from common import *
from analysis_math import split_indices, success_rule, temporal_bootstrap
from train import STRATEGIES
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import temporal_metrics

def independent(y, p):
    pr, re, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    d = {'f1_macro': float(f.mean()), 'accuracy': float(accuracy_score(y, p))}
    for j, k in enumerate(LABELS):
        d.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j])})
    return d

def independent_bootstrap(y, a, b, seed=20261026, reps=2000):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in range(3)]
    d = []

    def macro(y, p):
        c = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
        den = c.sum(0) + c.sum(1)
        return np.divide(2 * c.diagonal(), den, out=np.zeros(3), where=den > 0).mean()
    for _ in range(reps):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        d.append(np.mean([macro(y[ix], v[ix]) - macro(y[ix], u[ix]) for u, v in zip(a, b)]))
    return np.quantile(d, [0.025, 0.975])

def main():
    check_frozen()
    assert load(HERE / 'evaluated.json')['complete'] and load(HERE / 'publication.json')['published']
    rows = load(HERE / 'manifest.json')
    train, val, test = split_indices(rows)
    assert len(train) + len(val) == 8796 and len(test) == 688
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert all((rows[i]['group_id'] == 'V_6' for i in val))
    results = pd.read_csv(_resolve_model_artifact(HERE / 'results.csv'))
    assert len(results) == 12
    preds = {k: [] for k in ['central'] + STRATEGIES}
    canonical = None
    checks = []
    for seed in SEEDS:
        base = _load_model_result_array(HERE / f'seed_{seed}/central/vf_probabilities.npy').argmax(1)
        for strategy in ['central'] + STRATEGIES:
            out = HERE / f'seed_{seed}' / strategy
            f = pd.read_csv(_resolve_model_artifact(out / 'vf_predictions.csv'))
            p = _load_model_result_array(out / 'vf_probabilities.npy')
            a = _load_model_result_array(out / 'vf_attention.npy')
            sel = load(out / 'selected.json')
            assert digest(out / 'best.weights.h5') == sel['checkpoint_sha256'] and sel['epochs'] == 15
            key = f[['clip_id', 'y_true', 'section', 'start_s', 'end_s']]
            if canonical is None:
                canonical = key
            else:
                pd.testing.assert_frame_equal(key, canonical)
            assert f.clip_id.nunique() == 688 and p.shape == (688, 3) and np.isfinite(p).all()
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-06)
            np.testing.assert_allclose(p, f[['p_' + k for k in LABELS]], atol=1e-07)
            prediction = p.argmax(1)
            assert list(f.y_final) == [LABELS[k] for k in prediction]
            y = np.array([LABELS.index(k) for k in f.y_true])
            m = independent(y, prediction)
            saved = load(out / 'vf_metrics.json')
            row = results[(results.seed == seed) & (results.strategy == strategy)].iloc[0]
            for k, v in m.items():
                assert np.isclose(v, row[k], atol=1e-12) and np.isclose(v, saved['block'][k], atol=1e-12)
            corrected = int(((base != y) & (prediction == y)).sum())
            broken = int(((base == y) & (prediction != y)).sum())
            assert corrected == row['corrected'] == saved['changes']['corrected']
            assert broken == row['broken'] == saved['changes']['broken']
            assert corrected - broken == int((prediction == y).sum() - (base == y).sum())
            ev = temporal_metrics(f)
            for k in ['line', 'scrum']:
                for metric in ['true_events', 'predicted_events', 'matched_events', 'weighted_f1']:
                    assert np.isclose(ev[k][metric], saved['temporal'][k][metric])
            assert a.shape == (688, 9) and (a >= 0).all()
            np.testing.assert_allclose(a.sum(1), 1, atol=1e-06)
            if strategy == 'central':
                np.testing.assert_allclose(p, _load_model_result_array(CACHE / f'seed_{seed}/fixed/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
            elif strategy == 'late_mix':
                assert (a[:, 0] >= 0.5).all()
            else:
                np.testing.assert_allclose(a[:, 0], 0)
            preds[strategy].append(prediction)
            checks.append(dict(seed=seed, strategy=strategy, passed=True))
    means = results.groupby('strategy').mean(numeric_only=True)
    comparisons = load(HERE / 'comparisons.json')['comparisons']
    for strategy, c in comparisons.items():
        np.testing.assert_allclose(independent_bootstrap(y, preds['central'], preds[strategy]), c['stratified_bootstrap']['ci95'], atol=1e-12)
        temporal = temporal_bootstrap(y, preds['central'], preds[strategy], canonical.section.to_numpy(), canonical.start_s.to_numpy())
        np.testing.assert_allclose(temporal['ci95'], c['temporal_bootstrap']['ci95'], atol=1e-12)
        improvement = [independent(y, b)['f1_macro'] > independent(y, a)['f1_macro'] for a, b in zip(preds['central'], preds[strategy])]
        assert success_rule(means.loc['central'].to_dict(), means.loc[strategy].to_dict(), c['stratified_bootstrap']['ci95'], improvement, c['temporal_bootstrap']['ci95']) == c['evidence_rule']
    pilot = pd.read_csv(_resolve_model_artifact(HERE / 'pilot/results.csv')).set_index('strategy')
    yv = np.array([LABELS.index(rows[i]['tag']) for i in val])
    assert load(HERE / 'pilot/central/selected.json')['train_examples'] == len(train)
    assert digest(HERE / 'pilot/central/best.weights.h5') == load(HERE / 'embeddings/pilot/identity.json')['checkpoint_sha256']
    for strategy in ['central'] + STRATEGIES:
        p = _load_model_result_array(HERE / 'pilot' / strategy / 'validation_probabilities.npy')
        assert p.shape == (len(val), 3)
        for k, v in independent(yv, p.argmax(1)).items():
            assert np.isclose(v, pilot.loc[strategy, k], atol=1e-12)
    http = {}
    baseurl = 'http://127.0.0.1:5000'
    for file in ['index.html', 'timelines.png']:
        path = f'/static/reports/{HERE.name}/{file}'
        with urlopen(baseurl + path, timeout=15) as res:
            assert res.status == 200
            http[path] = res.status
    for strategy in ['central'] + STRATEGIES:
        frame = pd.read_csv(_resolve_model_artifact(HERE / f'seed_42/{strategy}/vf_predictions.csv'))
        for section in ['S01', 'S02', 'S03', 'S04']:
            rid = f'20261026_central_preserving_{strategy}_D2_VF_{section}_S42'
            data = load(ROOT / 'data/results' / f'result_{rid}.json')
            f = frame[frame.section == section]
            assert [w['play_type'] for w in data['window_predictions']] == list(f.y_final) and len(f) == 172
            assert np.isclose(data['window_predictions'][-1]['duration'], 4.2)
            for path in ['/viewer?result_id=' + rid, '/api/result/' + rid]:
                with urlopen(baseurl + path, timeout=15) as res:
                    assert res.status == 200
                    http[path] = 200
                    if path.startswith('/api/result'):
                        assert json.load(res)['results'] == data
    verification = dict(passed=True, prediction_checks=checks, pilot_verified=True, bootstrap_reproduced=True, http=http, no_thresholds=True, independent_test=False)
    save(HERE / 'verification.json', verification)
    backup = ROOT / 'model_backups' / f'{HERE.name}_verified.zip'
    tmp = backup.with_suffix('.tmp')
    files = list(_model_artifact_glob(HERE, '*.py', recursive=False)) + list(_model_artifact_glob(HERE, '*.json', recursive=False)) + list(_model_artifact_glob(HERE, '*.csv', recursive=False)) + [HERE / 'RESULTS.md'] + list(_model_artifact_glob(HERE, 'partitions/*.csv', recursive=False)) + list(_model_artifact_glob(HERE, 'seed_*/*/*', recursive=False)) + list(_model_artifact_glob(HERE, 'pilot/*/*', recursive=False)) + [HERE / 'pilot/results.csv', HERE / 'pilot/validation.json']
    files += [HERE / 'partitions/pilot_indices.json'] + [f for d in ['pilot'] + [f'seed_{s}' for s in SEEDS] for f in _model_artifact_glob(HERE / 'embeddings' / d, '*.json', recursive=False)]
    files = [f for f in dict.fromkeys(files) if _resolve_model_artifact(f).is_file() and f.suffix not in ['.lock'] and (f.name not in ['backup_complete.json', 'completion.json', 'status.json', 'process.json', 'run.json', 'pipeline_complete.json'])]
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(HERE).as_posix())
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        for f in files:
            assert hashlib.sha256(z.read(f.relative_to(HERE).as_posix())).hexdigest() == digest(f)
    tmp.replace(backup)
    save(HERE / 'backup_complete.json', dict(passed=True, path=str(backup), sha256=digest(backup), entries=len(files)))
    save(HERE / 'completion.json', dict(complete=True, verified=True, new_final_heads=9, pilot_heads=3, reused_centrals=3, pilot_central=1))
    save(HERE / 'status.json', dict(phase='complete', passed=[k for k, v in comparisons.items() if v['evidence_rule']['passed']]))
    print(json.dumps(dict(verified=True, evidence={k: v['evidence_rule'] for k, v in comparisons.items()}), indent=2), flush=True)
if __name__ == '__main__':
    main()
