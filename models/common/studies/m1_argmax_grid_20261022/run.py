"""Extend the original numeric threshold grid with an argmax option per class.

Frozen models, selection on development only, evaluation on held-out fold tests.
Mixed mode preserves all argmax decisions of that class, then thresholds the
other class on remaining blocks. Both numeric modes retain the original rule.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import os, sys, json, hashlib, gc, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
print('Loading numpy/pandas', flush=True)
import numpy as np
import pandas as pd
from models.common.comparison import metrics as m
from models.common.comparison import data as data_api
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
CACHE = HERE / 'probabilities'
CACHE.mkdir(exist_ok=True)
RUNS = ROOT / 'data/final_model_comparison_20260827/outputs/runs'
LABELS = np.array(['line', 'other', 'scrum'])
GRID = list(np.arange(0.1, 0.951, 0.05)) + ['argmax']

def sha(path):
    return hashlib.sha256(_resolve_model_artifact(Path(path)).read_bytes()).hexdigest()

def decide(p, line, scrum):
    a = p.argmax(1)
    if line == 'argmax' and scrum == 'argmax':
        return a
    if scrum == 'argmax':
        return np.where(a == 2, 2, np.where(p[:, 0] >= line, 0, 1))
    if line == 'argmax':
        return np.where(a == 0, 0, np.where(p[:, 2] >= scrum, 2, 1))
    return m.apply_class_thresholds(p, line=line, scrum=scrum)

class Evaluator:

    def __init__(self, f):
        self.f = f.reset_index(drop=True)
        self.p = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
        self.truth = {c: m.events_from_blocks(self.f, c, prediction_column='y_true', probability_column='p_' + c) for c in ['line', 'scrum']}
        self.ytrue = np.array([{'line': 0, 'other': 1, 'scrum': 2}[v] for v in f.y_true])
        self.groups = []
        for (group, section), g in self.f.groupby(['group_id', 'section'], sort=False):
            g = g.sort_values('block_index', kind='mergesort')
            self.groups.append((str(group), str(section), g.index.to_numpy(), g.block_index.to_numpy(), g.start_s.to_numpy(), g.end_s.to_numpy()))
        self.hours = sum((float(e.max()) for _, _, _, _, _, e in self.groups)) / 3600
        self.cache = {'line': {}, 'scrum': {}}

    def class_metrics(self, y, c):
        k = 0 if c == 'line' else 2
        mask = y == k
        key = mask.tobytes()
        if key in self.cache[c]:
            return self.cache[c][key]
        pr = []
        confidence = self.f['p_' + c].to_numpy()
        for group, sec, ids, blocks, starts, ends in self.groups:
            active = np.flatnonzero(mask[ids])
            if not len(active):
                continue
            chunks = np.split(active, np.flatnonzero(np.diff(blocks[active]) != 1) + 1)
            for positions in chunks:
                best = positions[np.argmax(confidence[ids[positions]])]
                pr.append(m.Event(group, sec, c, float(starts[positions].min()), float(ends[positions].max()), float((starts[best] + ends[best]) / 2), tuple(map(int, blocks[positions]))))
        tr = self.truth[c]
        pairs = m.match_events(tr, pr)
        matched = {j for _, j, _, _ in pairs}
        far = sum((j not in matched and (not any((t.group == p.group and t.section == p.section and (abs(t.anchor - p.anchor) <= 35) for t in tr))) for j, p in enumerate(pr)))
        tp = len(pairs)
        fp = len(pr) - tp
        fn = len(tr) - tp
        credit = sum((v[3] for v in pairs))
        result = dict(real=len(tr), tp=tp, fp=fp, fn=fn, candidates=len(pr), credit=credit, temporal_f1=2 * credit / (len(tr) + len(pr)) if tr or pr else 1, far_fp_per_hour=far / self.hours, precision=tp / len(pr) if pr else 0, recall=tp / len(tr) if tr else 0, f1=2 * tp / (len(tr) + len(pr)) if tr or pr else 1)
        self.cache[c][key] = result
        return result

    def evaluate(self, y):
        r = {}
        for c in ['line', 'scrum']:
            r.update({c + '_' + k: v for k, v in self.class_metrics(y, c).items()})
        r['macro_temporal_f1'] = (r['line_temporal_f1'] + r['scrum_temporal_f1']) / 2
        cf = np.bincount(3 * self.ytrue + y, minlength=9).reshape(3, 3)
        f1 = np.divide(2 * cf.diagonal(), cf.sum(0) + cf.sum(1), out=np.zeros(3), where=cf.sum(0) + cf.sum(1) > 0)
        r['tie_min_block_f1'] = min(f1[0], f1[2])
        r['tie_negative_far_fp_hour'] = -r['line_far_fp_per_hour'] - r['scrum_far_fp_per_hour']
        return r

    def verify(self, y):
        d = self.f.assign(y_final=LABELS[y])
        ref = m.temporal_metrics(d)
        got = self.evaluate(y)
        for c in ['line', 'scrum']:
            assert np.isclose(got[c + '_temporal_f1'], ref[c]['weighted_f1'], atol=1e-12)
            assert got[c + '_candidates'] == ref[c]['predicted_events']
            assert np.isclose(got[c + '_far_fp_per_hour'], ref[c]['far_false_positives_per_hour'])

def main():
    master = data_api.load_master_manifest()
    master_hash = sha(data_api.MASTER_MANIFEST)
    selection = []
    test_rows = []
    dev_rows = []
    files = {}
    predictions = []
    for i in range(1, 7):
        fold = f'PV{i:02d}'
        runid = f'PV-M1-H2-D2-{fold}-S42'
        folder = RUNS / runid / 'attempt_001'
        run = json.loads(_resolve_model_artifact(folder / 'run.json').read_text())
        for path, digest in run['artifacts'].items():
            assert sha(path) == digest
            files[path] = digest
        testpath = folder / 'predictions.csv'
        files[str(testpath)] = sha(testpath)
        test = pd.read_csv(_resolve_model_artifact(testpath))
        test['group_id'] = test.test_group
        dev = master[master.domain.eq('CV') & master.group_id.eq(run['development_group'])].reset_index(drop=True)
        assert len(dev) == run['n_dev'] and set(dev.block_id).isdisjoint(test.block_id)
        assert run['development_group'] != run['test_group'] and run['development_group'] not in run['train_groups']
        cache = CACHE / (fold + '_development.csv')
        if _resolve_model_artifact(cache).exists():
            d = pd.read_csv(_resolve_model_artifact(cache))
            assert list(d.block_id) == list(dev.block_id)
        else:
            import keras, joblib
            from models.common.comparison.models import _predict_keras, _geometric_fuse
            print(f"{fold}: infer development {run['development_group']}, {len(dev)} blocks from frozen model", flush=True)
            model = keras.models.load_model(_resolve_model_artifact(folder / 'dual.keras'), compile=False)
            forest = joblib.load(_resolve_model_artifact(folder / 'geometry_rf.joblib'))
            forest.n_jobs = 2
            dual = _predict_keras(model, dev, batch_size=64, dual=True)
            geom = forest.predict_proba(data_api.compact_matrix(dev, ('geometry',)))
            p = _geometric_fuse(dual, geom, float(run['model_metadata']['geometry_alpha']))
            sample = master.set_index('block_id').loc[test.block_id.iloc[:8]].reset_index()
            check = _predict_keras(model, sample, batch_size=64, dual=True)
            assert np.allclose(check, test[['p_dual_line', 'p_dual_other', 'p_dual_scrum']].iloc[:8], atol=2e-05), fold
            d = dev.copy()
            d['y_true'] = d.label
            for j, c in enumerate(LABELS):
                d['p_' + c] = p[:, j]
            d.to_csv(cache, index=False)
            del model, forest
            keras.backend.clear_session()
            data_api._load_resnet_path.cache_clear()
            gc.collect()
        de = Evaluator(d)
        te = Evaluator(test)
        original = run['thresholds']
        for ev in [de, te]:
            for line, scrum in [(original['line'], original['scrum']), ('argmax', 'argmax'), (0.35, 'argmax'), ('argmax', 0.35)]:
                ev.verify(decide(ev.p, line, scrum))
        best = None
        oldbest = None
        for line in GRID:
            for scrum in GRID:
                mode = 'both_argmax' if line == scrum == 'argmax' else 'line_argmax' if line == 'argmax' else 'scrum_argmax' if scrum == 'argmax' else 'numeric_numeric'
                dm = de.evaluate(decide(de.p, line, scrum))
                row = dict(fold=fold, development_group=run['development_group'], line_rule=line, scrum_rule=scrum, family=mode, **dm)
                dev_rows.append(row)
                key = tuple((dm[c] for c in ['macro_temporal_f1', 'tie_min_block_f1', 'tie_negative_far_fp_hour']))
                if best is None or key > best[0]:
                    best = (key, line, scrum, dm)
                if mode == 'numeric_numeric' and (oldbest is None or key > oldbest[0]):
                    oldbest = (key, line, scrum, dm)
                tm = te.evaluate(decide(te.p, line, scrum))
                test_rows.append(dict(fold=fold, test_group=run['test_group'], line_rule=line, scrum_rule=scrum, family=mode, **tm))
        assert np.isclose(oldbest[1], original['line']) and np.isclose(oldbest[2], original['scrum']), (fold, oldbest[1:3], original)
        for policy, line, scrum, devmetric in [('original_numeric', oldbest[1], oldbest[2], oldbest[3]), ('expanded_grid', best[1], best[2], best[3]), ('argmax', 'argmax', 'argmax', de.evaluate(decide(de.p, 'argmax', 'argmax')))]:
            y = decide(te.p, line, scrum)
            tm = te.evaluate(y)
            selection.append(dict(fold=fold, test_group=run['test_group'], development_group=run['development_group'], policy=policy, line_rule=line, scrum_rule=scrum, development_macro_temporal_f1=devmetric['macro_temporal_f1'], **tm))
            if policy == 'expanded_grid':
                predictions.append(test.assign(y_final=LABELS[y], selection_policy=policy, fold=fold))
            if policy == 'original_numeric':
                previous = json.loads(_resolve_model_artifact(folder / 'metrics.json').read_text())
                assert np.isclose(tm['macro_temporal_f1'], previous['temporal']['weighted_f1_macro'])
        print(f"{fold}: selected line={best[1]}, scrum={best[2]}, dev F1={best[3]['macro_temporal_f1']:.6f}", flush=True)
        pd.DataFrame(selection).to_csv(OUT / 'selected_test_metrics.csv', index=False)
    pd.DataFrame(dev_rows).to_csv(OUT / 'development_grid.csv', index=False)
    pd.DataFrame(test_rows).to_csv(OUT / 'test_grid_descriptive.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'selected_test_predictions.csv.gz', index=False, compression='gzip')
    table = pd.DataFrame(selection)
    totals = []
    for policy, g in table.groupby('policy', sort=False):
        row = dict(policy=policy, mean_macro_temporal_f1=float(g.macro_temporal_f1.mean()))
        for c in ['line', 'scrum']:
            tp, fp, fn = (int(g[c + '_' + k].sum()) for k in ['tp', 'fp', 'fn'])
            row.update({c + '_tp': tp, c + '_fp': fp, c + '_fn': fn, c + '_precision': tp / (tp + fp), c + '_recall': tp / (tp + fn), c + '_f1': 2 * tp / (2 * tp + fp + fn), c + '_mean_temporal_f1': float(g[c + '_temporal_f1'].mean())})
        totals.append(row)
    pd.DataFrame(totals).to_csv(OUT / 'aggregate_metrics.csv', index=False)
    assert sha(data_api.MASTER_MANIFEST) == master_hash
    assert all((sha(p) == h for p, h in files.items()))
    (HERE / 'verification.json').write_text(json.dumps(dict(passed=True, grid_per_class=[float(v) if v != 'argmax' else v for v in GRID], combinations_per_fold=len(GRID) ** 2, selection='Development only, original tie breakers', mixed_policy='Preserve the argmax class, threshold the other class on remaining blocks', original_numeric_optima_reproduced=True, original_test_metrics_reproduced=True, fast_metrics_checked_against_original=True, no_training=True, no_source_modifications=True, source_sha256=files, master_sha256=master_hash), indent=2), encoding='utf-8')
    print(table[['fold', 'policy', 'line_rule', 'scrum_rule', 'line_tp', 'line_fp', 'line_fn', 'scrum_tp', 'scrum_fp', 'scrum_fn', 'macro_temporal_f1']].to_string(index=False), flush=True)
    print(pd.DataFrame(totals).to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
