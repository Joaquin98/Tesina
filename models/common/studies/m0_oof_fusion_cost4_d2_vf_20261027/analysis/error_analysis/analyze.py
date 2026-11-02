"""Exploratory audit of VF errors and fixed decision rules, not a new test."""
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
R = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
P = R / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
OUT = experiment_directory(__file__)
BASE = R / 'data/central_recall_cost_d2_vf_20261027'
LABELS = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]

def metrics(y, p):
    pr, re, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    d = dict(accuracy=float(accuracy_score(y, p)), f1_macro=float(f.mean()))
    for j, k in enumerate(LABELS):
        d.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(cm[j, j]), k + '_fn': int(cm[j].sum() - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
    return d

def rules(p, f):
    n = len(p)
    v = p.argmax(2)
    line = (v == 0).sum(1)
    scrum = (v == 2).sum(1)
    r = {'fusion': f, 'central': v[:, 0]}
    for j in range(1, 9):
        r[f'vista_{j}'] = v[:, j]
    r['maximo_global'] = p.reshape(n, 27).argmax(1) % 3
    q = np.take_along_axis(p, v[:, :, None], axis=2)[:, :, 0]
    z = np.where(v != 1, q, -np.inf)
    winner = z.argmax(1)
    r['evento_maximo'] = np.where((v != 1).any(1), v[np.arange(n), winner], 1)
    r['votos_evento'] = np.where(line + scrum == 0, 1, np.where(scrum >= line, 2, 0))
    r['promedio_probabilidades'] = p.mean(1).argmax(1)
    for k in [1, 2, 3]:
        event = np.where(scrum >= line, 2, 0)
        r[f'rescate_other_{k}_votos'] = np.where((f == 1) & (line + scrum >= k), event, f)
    return (r, v, line, scrum)

def main():
    assert json.loads(_resolve_model_artifact(P / 'completion.json').read_text())['complete']
    df = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    assert len(df) == 688
    assert df.clip_id.is_unique
    y = np.array([LABELS.index(x) for x in df.tag])
    assert np.bincount(y).tolist() == [68, 581, 39]
    allrows = []
    summaries = []
    predictions = {}
    perseed = {}
    hashes = {}
    for seed in SEEDS:
        files = [P / f'seed_{seed}/vf_views.npy', P / f'seed_{seed}/vf_fusion.npy']
        hashes.update({str(p): hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest() for p in files})
        p = _load_model_result_array(files[0])
        fprob = _load_model_result_array(files[1])
        assert p.shape == (688, 9, 3) and fprob.shape == (688, 3)
        np.testing.assert_allclose(p.sum(2), 1, atol=1e-05)
        np.testing.assert_allclose(fprob.sum(1), 1, atol=1e-05)
        np.testing.assert_allclose(p[:, 0], _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        f = fprob.argmax(1)
        assert [LABELS[j] for j in f] == df[f'fusion_{seed}'].tolist()
        rr, v, line, scrum = rules(p, f)
        predictions[seed] = rr
        perseed[seed] = (v, line, scrum, f)
        assert [LABELS[j] for j in v[:, 0]] == df[f'central_{seed}'].tolist()
        for name, pred in rr.items():
            m = metrics(y, pred)
            event = y != 1
            errors = f != y
            m.update(seed=seed, rule=name, corrected_event=int((event & errors & (pred == y)).sum()), broken_event=int((event & ~errors & (pred != y)).sum()), corrected_all=int((errors & (pred == y)).sum()), broken_all=int((~errors & (pred != y)).sum()))
            summaries.append(m)
        for i in range(len(y)):
            correct = np.flatnonzero(v[i] == y[i])
            row = dict(seed=seed, clip_id=df.clip_id.iloc[i], section=df.section.iloc[i], start_s=df.start_s.iloc[i], end_s=df.end_s.iloc[i], human=LABELS[y[i]], central=LABELS[v[i, 0]], fusion=LABELS[f[i]], fusion_error=bool(f[i] != y[i]), central_error=bool(v[i, 0] != y[i]), correct_view_exists=bool(len(correct)), correct_views=','.join(map(str, correct.tolist())), correct_view_count=len(correct), line_votes=int(line[i]), scrum_votes=int(scrum[i]), other_votes=int((v[i] == 1).sum()), central_probabilities=' / '.join((f'{x:.4f}' for x in p[i, 0])), fusion_probabilities=' / '.join((f'{x:.4f}' for x in fprob[i])))
            for j in range(9):
                row[f'vista_{j}'] = LABELS[v[i, j]]
            for name, pred in rr.items():
                row[f'regla_{name}'] = LABELS[pred[i]]
            allrows.append(row)
    detail = pd.DataFrame(allrows)
    detail.to_csv(OUT / 'todos_bloques_y_semillas.csv', index=False)
    event_errors = detail[detail.human.isin(['line', 'scrum']) & detail.fusion_error]
    event_errors.to_csv(OUT / 'errores_line_scrum.csv', index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(OUT / 'reglas_por_semilla.csv', index=False)
    mean = summ.groupby('rule').mean(numeric_only=True)
    mean.to_csv(OUT / 'reglas_promedio.csv')
    for seed in SEEDS:
        saved = pd.read_csv(_resolve_model_artifact(P / 'results.csv'))
        sr = saved[(saved.seed == seed) & (saved.variant == 'fusion')].iloc[0]
        mr = metrics(y, predictions[seed]['fusion'])
        for k in mr:
            np.testing.assert_allclose(mr[k], sr[k], atol=1e-12)
    result = {'label_scope': '688 annotated VF blocks; individual block not unique rugby event', 'seeds': SEEDS, 'hashes': hashes, 'n_rows': len(detail), 'n_event_errors': len(event_errors), 'recovered_by_view': int(event_errors.correct_view_exists.sum()), 'scope': 'predictions repeated across seeds, not independent examples; all rules inspected in observed VF are exploratory, no thresholds tuned', 'counts_by_seed_class': event_errors.groupby(['seed', 'human']).agg(errors=('fusion_error', 'size'), any_correct=('correct_view_exists', 'sum')).reset_index().to_dict(orient='records'), 'rule_means': mean.to_dict(orient='index')}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
if __name__ == '__main__':
    main()
