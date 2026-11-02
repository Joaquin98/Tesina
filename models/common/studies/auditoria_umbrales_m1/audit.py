"""Read-only reproduction. Writes exclusively beside this script. No fitting.

python models/common/studies/auditoria_umbrales_m1/audit.py core
python models/common/studies/auditoria_umbrales_m1/audit.py development --fold 1
python models/common/studies/auditoria_umbrales_m1/audit.py verify
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse, hashlib, itertools, json, os, platform, sys, time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.dont_write_bytecode = True
OUT = experiment_directory(__file__)
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.optimize import linprog
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
from models.common.comparison import metrics as m
from models.common.comparison.protocol import LABELS, LABEL_TO_ID, select_run
from models.common.comparison.data import split_for_run
EXP = ROOT / 'data/final_model_comparison_20260827'
RUNS = EXP / 'outputs/runs'
PRIOR = ROOT / 'data/analisis_errores_partido'

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def runpath(i):
    return RUNS / f'PV-M1-H2-D2-PV{i:02d}-S42/attempt_001'

def protected():
    paths = list(_model_artifact_glob(EXP, '*.py', recursive=False)) + [EXP / 'outputs/master_examples.csv', PRIOR / 'segments_all.csv', PRIOR / 'temporal_metrics.csv', PRIOR / 'expanded_census/table_argmax_final_paired.csv', PRIOR / 'expanded_census/table_decision_changes.csv', PRIOR / 'expanded_census/REPORT.md', _resolve_model_artifact(PRIOR / 'expanded_census/write_report.py'), PRIOR / 'informe_html/informe_completo.html', ROOT / 'docs/PLAN_EXPERIMENTAL_AFINADO_CINCO_MODELOS_20260827.md', ROOT / 'docs/PROTOCOLO_MEDICION_Y_REGISTRO_AFINADO_20260827.md']
    paths += list(_model_artifact_glob(ROOT / 'docs/thesis', '*.tex', recursive=True))
    paths += [p for i in range(1, 7) for p in runpath(i).iterdir() if _resolve_model_artifact(p).is_file()]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}

def frame_for(i):
    frame = pd.read_csv(_resolve_model_artifact(runpath(i) / 'predictions.csv'), float_precision='round_trip')
    frame['group_id'] = frame.test_group
    return frame

def block_rows(frame, split, group):
    rows, summary = ([], [])
    for decision in ('y_argmax', 'y_final'):
        y, p = (frame.y_true, frame[decision])
        pr, re, f, s = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
        for j, c in enumerate(LABELS):
            tp = int(((y == c) & (p == c)).sum())
            fp = int(((y != c) & (p == c)).sum())
            fn = int(((y == c) & (p != c)).sum())
            rows.append(dict(split=split, group=group, decision=decision, label=c, support=int(s[j]), tp=tp, fp=fp, fn=fn, precision=pr[j], recall=re[j], f1=f[j]))
        a = y == frame.y_argmax
        b = y == frame.y_final
        summary.append(dict(split=split, group=group, decision=decision, n=len(frame), errors=int((y != p).sum()), accuracy=accuracy_score(y, p), f1_macro=float(np.mean(f)), other_false_accepts=int(((y == 'other') & (p != 'other')).sum()), kept_correct=int((a & b).sum()), kept_wrong=int((~a & ~b).sum()), corrected=int((~a & b).sum()), spoiled=int((a & ~b).sum()), changes=int((frame.y_argmax != frame.y_final).sum())))
    return (rows, summary)

def temporal_rows(frame, split, group):
    rows, full = ([], {})
    for decision in ('y_argmax', 'y_final'):
        temp = frame.copy()
        temp['y_final'] = frame[decision].to_numpy()
        metrics = m.temporal_metrics(temp)
        full[decision] = metrics
        for c in ('line', 'scrum'):
            v = metrics[c]
            row = dict(split=split, group=group, decision=decision, label=c, **{k: v[k] for k in v if not isinstance(v[k], dict)})
            row.update({f'f1_at_{sec}s': score for sec, score in v['f1_at_seconds'].items()})
            row.update({f'f1_iou_{iou}': score for iou, score in v['f1_at_iou'].items()})
            rows.append(row)
    return (rows, full)

def representable(frame, *, return_solution=False):
    p = m.normalize_probabilities(frame[[f'p_{c}' for c in LABELS]].to_numpy())
    a = []
    b = []
    for v, c in zip(p, frame.y_argmax):
        l, o, s = v
        if c == 'line':
            a.extend([[1, 0, 0], [s, -l, 0]])
            b.extend([l, 0])
        elif c == 'scrum':
            a.extend([[0, 1, 0], [-s, l, 1]])
            b.extend([s, 0])
        else:
            a.extend([[-1, 0, 1], [0, -1, 1]])
            b.extend([-l, -s])
    sol = linprog([0, 0, -1], A_ub=a, b_ub=b, bounds=[(1e-08, 1), (1e-08, 1), (0, 1)], method='highs')
    result = dict(success=sol.success, status=sol.message, strict_feasible=bool(sol.success and sol.x[2] > 1e-08))
    if return_solution:
        result['solution'] = sol.x.tolist() if sol.success else None
    return result

def core():
    if not _resolve_model_artifact(OUT / 'protected_before.json').exists():
        dump('protected_before.json', protected())
    dump('environment.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    master = pd.read_csv(_resolve_model_artifact(EXP / 'outputs/master_examples.csv'))
    previous = pd.read_csv(_resolve_model_artifact(PRIOR / 'segments_all.csv'), float_precision='round_trip').query("model == 'M1'")
    br = []
    bs = []
    tr = []
    checks = []
    frames = []
    tjson = {}
    feasibility = {}
    changed = []
    mechanisms = []
    splits = []
    grid = []
    for i in range(1, 7):
        path = runpath(i)
        meta = json.loads(_resolve_model_artifact(path / 'run.json').read_text(encoding='utf-8'))
        saved = json.loads(_resolve_model_artifact(path / 'metrics.json').read_text(encoding='utf-8'))
        f = frame_for(i)
        train, dev, test = split_for_run(master, select_run(meta['run_id']))
        t = meta['thresholds']
        probs = f[[f'p_{c}' for c in LABELS]].to_numpy()
        pred = np.asarray(LABELS)[m.apply_class_thresholds(probs, line=t['line'], scrum=t['scrum'])]
        fused = np.exp((1 - meta['model_metadata']['geometry_alpha']) * np.log(np.clip(f[[f'p_dual_{c}' for c in LABELS]].to_numpy(), 1e-07, 1)) + meta['model_metadata']['geometry_alpha'] * np.log(np.clip(f[[f'p_geometry_{c}' for c in LABELS]].to_numpy(), 1e-07, 1)))
        fused = m.normalize_probabilities(fused)
        old = previous.query('run_id == @meta["run_id"]') if False else previous[previous.run_id == meta['run_id']]
        old = old.set_index('block_id').loc[f.block_id]
        checks.append(dict(group=meta['test_group'], n=len(f), threshold_line=t['line'], threshold_scrum=t['scrum'], alpha=meta['model_metadata']['geometry_alpha'], finite=bool(np.isfinite(probs).all()), prob_min=float(probs.min()), prob_max=float(probs.max()), max_sum_error=float(np.max(np.abs(probs.sum(1) - 1))), normalization_max_delta=float(np.max(np.abs(m.normalize_probabilities(probs) - probs))), argmax_mismatches=int((np.asarray(LABELS)[probs.argmax(1)] != f.y_argmax).sum()), final_mismatches=int((pred != f.y_final).sum()), fusion_max_delta=float(np.max(np.abs(fused - probs))), test_order_matches=bool(list(f.block_id) == list(test.block_id)), truth_mismatches=int((f.y_true.to_numpy() != test.label.to_numpy()).sum()), prior_label_mismatches=int((f[['y_true', 'y_argmax', 'y_final']].to_numpy() != old[['y_true', 'y_argmax', 'y_final']].to_numpy()).sum()), prior_prob_max_delta=float(np.max(np.abs(probs - old[[f'p_{c}' for c in LABELS]].to_numpy()))), verifier_nonempty=int(f.line_verifier_score.notna().sum()), saved_block_equals_argmax=bool(saved['block'] == m.block_metrics(f.y_true.map(LABEL_TO_ID).to_numpy(), probs)), artifact_hashes_match=all((sha(Path(p)) == v for p, v in meta['artifacts'].items()))))
        r, s = block_rows(f, 'test', meta['test_group'])
        br.extend(r)
        bs.extend(s)
        r, j = temporal_rows(f, 'test', meta['test_group'])
        tr.extend(r)
        tjson[meta['test_group']] = j
        checks[-1]['temporal_saved_max_delta'] = nested_delta(saved['temporal'], j['y_final'])
        for col in ('block_index', 'section', 'start_s', 'end_s', 'duration'):
            checks[-1][f'{col}_matches_manifest'] = bool(np.array_equal(f[col].to_numpy(), test[col].to_numpy()))
        splits.append(dict(fold=meta['fold'], test_group=meta['test_group'], development_group=meta['development_group'], train_groups=','.join(meta['train_groups']), n_train=len(train), n_dev=len(dev), n_test=len(test), train_ch=int((train.domain == 'CH').sum()), train_cv=int((train.domain == 'CV').sum()), metadata_sizes_match=bool((len(train), len(dev), len(test)) == (meta['n_train'], meta['n_dev'], meta['n_test'])), block_overlap=0, source_overlap=bool(set(train.source_group) & (set(dev.source_group) | set(test.source_group)))))
        fc = f[f.y_argmax != f.y_final].copy()
        fc['threshold_line'] = t['line']
        fc['threshold_scrum'] = t['scrum']
        fc['line_score'] = fc.p_line / t['line']
        fc['scrum_score'] = fc.p_scrum / t['scrum']
        fc['argmax_correct'] = fc.y_argmax == fc.y_true
        fc['final_correct'] = fc.y_final == fc.y_true
        changed.append(fc)
        for (truth, a, b), chunk in fc.groupby(['y_true', 'y_argmax', 'y_final']):
            mechanisms.append(dict(group=meta['test_group'], truth=truth, argmax=a, final=b, n=len(chunk)))
        values = np.arange(0.1, 0.951, 0.05)
        match_counts = [int((np.asarray(LABELS)[m.apply_class_thresholds(probs, line=l, scrum=s)] == f.y_argmax).sum()) for l, s in itertools.product(values, values)]
        grid.append(dict(group=meta['test_group'], grid_candidates=len(match_counts), exact_argmax_candidates=match_counts.count(len(f)), minimum_disagreements_with_argmax=len(f) - max(match_counts)))
        feasibility[meta['test_group']] = representable(f)
        frames.append(f)
    allf = pd.concat(frames, ignore_index=True)
    r, s = block_rows(allf, 'test', 'ALL')
    br.extend(r)
    bs.extend(s)
    r, j = temporal_rows(allf, 'test', 'ALL_POOLED')
    tr.extend(r)
    tjson['ALL_POOLED'] = j
    for name, rows in [('block_per_class.csv', br), ('block_summary.csv', bs), ('temporal.csv', tr), ('integrity_checks.csv', checks), ('splits.csv', splits), ('transitions.csv', mechanisms), ('argmax_grid_representability.csv', grid)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False)
    pd.concat(changed, ignore_index=True).to_csv(OUT / 'all_174_changes.csv', index=False)
    dump('temporal_full.json', tjson)
    dump('argmax_continuous_feasibility.json', feasibility)
    synthetic = np.array([[0.34, 0.33, 0.33], [0.48, 0.49, 0.03], [0.33, 0.33, 0.34], [0.03, 0.49, 0.48]])
    sf = pd.DataFrame(synthetic, columns=['p_line', 'p_other', 'p_scrum'])
    sf['y_argmax'] = np.asarray(LABELS)[synthetic.argmax(1)]
    sf.to_csv(OUT / 'argmax_nonrepresentable_counterexample.csv', index=False)
    dump('counterexample_feasibility.json', representable(sf))
    print(pd.DataFrame(bs).to_string(index=False), flush=True)
    print('Temporal macro by match:', {g: {d: v['weighted_f1_macro'] for d, v in ds.items()} for g, ds in tjson.items()}, flush=True)

def nested_delta(a, b):
    if isinstance(a, dict):
        return max([nested_delta(a[k], b[k]) for k in a] or [0])
    if isinstance(a, list):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if a is None or isinstance(a, str):
        return 0 if a == b else float('inf')
    return abs(float(a) - float(b))

def development(i):
    import keras, joblib
    from models.common.comparison.models import _predict_keras, _geometric_fuse, _choose_alpha
    from models.common.comparison.data import load_yolo, geometry_statistics
    path = runpath(i)
    meta = json.loads(_resolve_model_artifact(path / 'run.json').read_text(encoding='utf-8'))
    master = pd.read_csv(_resolve_model_artifact(EXP / 'outputs/master_examples.csv'))
    train, dev, test = split_for_run(master, select_run(meta['run_id']))
    print('RECOVER', i, flush=True)
    cache = OUT / f'development_PV{i:02d}.npz'
    if _resolve_model_artifact(cache).exists():
        z = _load_model_result_array(cache)
        dual = z['dual']
        rf = z['geometry']
        p = z['probabilities']
        validation = json.loads(_resolve_model_artifact(OUT / f'verification_PV{i:02d}.json').read_text(encoding='utf-8'))
    else:
        model = keras.models.load_model(_resolve_model_artifact(path / 'dual.keras'), compile=False)
        forest = joblib.load(_resolve_model_artifact(path / 'geometry_rf.joblib'))
        assert list(forest.classes_) == [0, 1, 2]
        geom = np.asarray([geometry_statistics(load_yolo(row)) for row in dev.itertuples(index=False)], dtype=np.float32)
        rf = forest.predict_proba(geom)
        dual = _predict_keras(model, dev, batch_size=meta['batch_size'], dual=True)
        p = _geometric_fuse(dual, rf, meta['model_metadata']['geometry_alpha'])
        idx = np.unique(np.linspace(0, len(test) - 1, 24, dtype=int))
        sub = test.iloc[idx].reset_index(drop=True)
        td = _predict_keras(model, sub, batch_size=meta['batch_size'], dual=True)
        tg = forest.predict_proba(np.asarray([geometry_statistics(load_yolo(row)) for row in sub.itertuples(index=False)], dtype=np.float32))
        tf = _geometric_fuse(td, tg, meta['model_metadata']['geometry_alpha'])
        saved = frame_for(i).iloc[idx]
        validation = dict(sample_test_rows=len(idx), dual_test_max_delta=float(np.max(np.abs(td - saved[[f'p_dual_{c}' for c in LABELS]].to_numpy()))), geometry_test_max_delta=float(np.max(np.abs(tg - saved[[f'p_geometry_{c}' for c in LABELS]].to_numpy()))), fused_test_max_delta=float(np.max(np.abs(tf - saved[[f'p_{c}' for c in LABELS]].to_numpy()))), alpha_saved=meta['model_metadata']['geometry_alpha'], alpha_recovered=_choose_alpha(dev.label.map(LABEL_TO_ID).to_numpy(), dual, rf, geometric=True), forest_classes=forest.classes_.tolist(), keras=keras.__version__, joblib=joblib.__version__)
        np.savez_compressed(cache, probabilities=p, dual=dual, geometry=rf, block_id=dev.block_id.to_numpy(dtype=str))
        dump(f'verification_PV{i:02d}.json', validation)
    frame = dev.copy()
    frame['y_true'] = dev.label
    frame['y_argmax'] = np.asarray(LABELS)[p.argmax(1)]
    for j, c in enumerate(LABELS):
        frame[f'p_{c}'] = p[:, j]
    t = meta['thresholds']
    frame['y_final'] = np.asarray(LABELS)[m.apply_class_thresholds(p, line=t['line'], scrum=t['scrum'])]
    frame[['block_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'y_true', 'y_argmax', 'y_final', 'p_line', 'p_other', 'p_scrum']].to_csv(OUT / f'development_PV{i:02d}.csv', index=False)
    rows = []
    original = m.temporal_metrics
    grid = np.arange(0.1, 0.951, 0.05)

    def capture(candidate):
        result = original(candidate)
        k = len(rows)
        l, s = list(itertools.product(grid, grid))[k]
        f = f1_score(candidate.y_true.map(LABEL_TO_ID), candidate.y_final.map(LABEL_TO_ID), labels=np.arange(3), average=None, zero_division=0)
        rows.append(dict(line=l, scrum=s, weighted_f1_macro=result['weighted_f1_macro'], min_target_block_f1=min(f[0], f[2]), negative_far_fp_hour=-result['line']['far_false_positives_per_hour'] - result['scrum']['far_false_positives_per_hour'], errors=int((candidate.y_true != candidate.y_final).sum())))
        if len(rows) % 54 == 0:
            print('SEARCH', i, len(rows), flush=True)
        return result
    m.temporal_metrics = capture
    try:
        selected = m.threshold_search(frame, p)
    finally:
        m.temporal_metrics = original
    pd.DataFrame(rows).to_csv(OUT / f'development_candidates_PV{i:02d}.csv', index=False)
    b, s = block_rows(frame, 'development', meta['fold'])
    tr, j = temporal_rows(frame, 'development', meta['fold'])
    pd.DataFrame(b).to_csv(OUT / f'dev_block_per_class_PV{i:02d}.csv', index=False)
    pd.DataFrame(s).to_csv(OUT / f'dev_block_summary_PV{i:02d}.csv', index=False)
    pd.DataFrame(tr).to_csv(OUT / f'dev_temporal_PV{i:02d}.csv', index=False)
    keys = [(r['weighted_f1_macro'], r['min_target_block_f1'], r['negative_far_fp_hour']) for r in rows]
    best = max(keys)
    dump(f'development_search_PV{i:02d}.json', dict(saved_thresholds=t, reselected=selected, same_thresholds=bool(t['line'] == selected['line'] and t['scrum'] == selected['scrum']), temporal_comparison=j, full_key_ties=keys.count(best), primary_ties=sum((k[0] == best[0] for k in keys)), recovery_validation=validation, argmax_representability=representable(frame)))
    print('DONE', i, json.dumps(validation), selected['line'], selected['scrum'], j['y_argmax']['weighted_f1_macro'], j['y_final']['weighted_f1_macro'], flush=True)

def verify():
    before = json.loads(_resolve_model_artifact(OUT / 'protected_before.json').read_text(encoding='utf-8'))
    now = {str(p): sha(ROOT / p) for p in before}
    changed = [p for p in before if before[p] != now[p]]
    source = json.loads(_resolve_model_artifact(OUT / 'source_snapshot.json').read_text(encoding='utf-8')) if _resolve_model_artifact(OUT / 'source_snapshot.json').exists() else {}
    changed_source = [p for p in source if sha(ROOT / p) != source[p]]
    dump('protected_verification.json', dict(files_checked=len(before), changed=changed, all_unchanged=not changed, source_snapshot_files=len(source), source_snapshot_changed=changed_source))
    assert not changed and (not changed_source), (changed, changed_source)
    print('Protected files unchanged:', len(before))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['core', 'development', 'verify'])
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.phase == 'core':
        core()
    elif args.phase == 'development':
        development(args.fold)
    else:
        verify()
