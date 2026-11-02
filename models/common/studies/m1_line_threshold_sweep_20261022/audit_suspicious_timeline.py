"""Independent audit of raw M1 predictions, without the experiment metric helpers."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
OUT = HERE / 'audit_20261022'
OUT.mkdir(exist_ok=True)
BASE = ROOT / 'data/final_model_comparison_20260827/outputs/runs'
LABELS = np.array(['line', 'other', 'scrum'])

def decision(f, line, scrum):
    p = f[['p_line', 'p_other', 'p_scrum']].to_numpy().clip(1e-08)
    p = p / p.sum(1, keepdims=True)
    y = np.full(len(f), 'other', dtype='<U5')
    ls = p[:, 0] / line
    ss = p[:, 2] / scrum
    y[(p[:, 0] >= line) & (ls >= ss)] = 'line'
    y[(p[:, 2] >= scrum) & (ss > ls)] = 'scrum'
    return y

def events(f, label, col):
    result = []
    for section, s in f.groupby('section', sort=False):
        s = s.sort_values('block_index', kind='stable')
        selected = s[s[col] == label]
        chunks = []
        cur = []
        last = None
        for r in selected.to_dict('records'):
            if last is not None and r['block_index'] != last + 1:
                chunks.append(cur)
                cur = []
            cur.append(r)
            last = r['block_index']
        if cur:
            chunks.append(cur)
        for chunk in chunks:
            start = chunk[0]['start_s']
            end = chunk[-1]['end_s']
            peak = max(chunk, key=lambda r: r['p_' + label])
            anchor = (start + end) / 2 if col == 'y_true' else (peak['start_s'] + peak['end_s']) / 2
            result.append(dict(section=section, start=start, end=end, anchor=anchor))
    return result

def evaluate(f, label):
    t = events(f, label, 'y_true')
    p = events(f, label, 'audit_prediction')
    distance = np.full((len(t), len(p)), np.inf)
    for i, a in enumerate(t):
        for j, b in enumerate(p):
            d = abs(a['anchor'] - b['anchor'])
            if a['section'] == b['section'] and d <= 35:
                distance[i, j] = d
    credit = np.maximum(0, 1 - distance / 35)
    pairs = []
    if len(t) and len(p):
        ii, jj = linear_sum_assignment(-credit)
        pairs = [(int(i), int(j), float(distance[i, j]), float(credit[i, j])) for i, j in zip(ii, jj) if np.isfinite(distance[i, j])]
    n = len(pairs)
    weighted = sum((v[3] for v in pairs))
    return (dict(real=len(t), tp=n, fp=len(p) - n, fn=len(t) - n, temporal_f1=2 * weighted / (len(t) + len(p)) if len(t) + len(p) else 1), t, p, pairs)
prior = json.loads(_resolve_model_artifact(HERE / 'validation.json').read_text(encoding='utf-8'))
timeline = json.loads(_resolve_model_artifact(HERE / 'timeline_data.json').read_text(encoding='utf-8'))
rows = []
checks = []
sources = {}
event_tables = []
for n in range(1, 7):
    folder = BASE / f'PV-M1-H2-D2-PV0{n}-S42/attempt_001'
    raw = folder / 'predictions.csv'
    meta = json.loads(_resolve_model_artifact(folder / 'run.json').read_text())
    sources[str(raw)] = hashlib.sha256(_resolve_model_artifact(raw).read_bytes()).hexdigest()
    assert sources[str(raw)] == prior['source_hashes'][str(raw)]
    f = pd.read_csv(_resolve_model_artifact(raw))
    f['audit_prediction'] = decision(f, **{k: meta['thresholds'][k] for k in ['line', 'scrum']})
    assert np.array_equal(f.audit_prediction, f.y_final)
    left = f[['p_dual_line', 'p_dual_other', 'p_dual_scrum']].to_numpy()
    right = f[['p_geometry_line', 'p_geometry_other', 'p_geometry_scrum']].to_numpy()
    alpha = meta['model_metadata']['geometry_alpha']
    reconstructed = np.exp((1 - alpha) * np.log(np.clip(left, 1e-07, 1)) + alpha * np.log(np.clip(right, 1e-07, 1)))
    reconstructed = reconstructed.clip(1e-08)
    reconstructed /= reconstructed.sum(1, keepdims=True)
    fusion_error = float(np.max(np.abs(reconstructed - f[['p_line', 'p_other', 'p_scrum']].to_numpy())))
    assert fusion_error < 1e-06
    historical = json.loads(_resolve_model_artifact(folder / 'metrics.json').read_text())
    class_results = {}
    for c in ['line', 'scrum']:
        result, t, p, pairs = evaluate(f, c)
        class_results[c] = result
        for key in ['real', 'tp']:
            assert result[key] == historical['temporal'][c]['true_events' if key == 'real' else 'matched_events']
        assert np.isclose(result['temporal_f1'], historical['temporal'][c]['weighted_f1'], atol=1e-12)
        if c == 'line':
            for sec in f.section.unique():
                jt = timeline[f'V_{n}']['sections'][sec]['line']
                actual_t = [e for e in t if e['section'] == sec]
                actual_p = [e for e in p if e['section'] == sec]
                assert len(actual_t) == len(jt['truth']) and len(actual_p) == len(jt['pred']['original'])
                for a, b in zip(actual_t, jt['truth']):
                    assert all((np.isclose(a[k], b[k], atol=0.001) for k in ['start', 'end', 'anchor']))
                for a, b in zip(actual_p, jt['pred']['original']):
                    assert all((np.isclose(a[k], b[k], atol=0.001) for k in ['start', 'end', 'anchor']))
            rows.append(dict(match=f'V_{n}', **result, line_threshold=meta['thresholds']['line'], sensitivity=result['tp'] / result['real']))
    checks.append(dict(match=f'V_{n}', raw_decisions_reproduced=True, historical_event_metrics_reproduced=True, timeline_intervals_reproduced=True, probability_hash_unchanged=True, fusion_max_abs_error=fusion_error))
    if n == 4:
        section = f[f.section == 'S03'].copy()
        summary = []
        for threshold in [0.7, 0.65, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
            section['audit_prediction'] = decision(section, threshold, meta['thresholds']['scrum'])
            result, *_ = evaluate(section, 'line')
            summary.append(dict(rule='thresholds', threshold=threshold, **result))
        section['audit_prediction'] = LABELS[section[['p_line', 'p_other', 'p_scrum']].to_numpy().argmax(1)]
        result, *_ = evaluate(section, 'line')
        summary.append(dict(rule='argmax', threshold=None, **result))
        pd.DataFrame(summary).to_csv(OUT / 'V4_S03_independent_thresholds.csv', index=False)
        truth = events(section, 'line', 'y_true')
        per_event = []
        for i, t in enumerate(truth, 1):
            overlapping = section[(section.start_s < t['end']) & (section.end_s > t['start'])]
            near = section[abs((section.start_s + section.end_s) / 2 - t['anchor']) <= 35]
            per_event.append(dict(event=i, **t, max_p_line_on_truth=float(overlapping.p_line.max()), max_p_line_near_35s=float(near.p_line.max())))
        pd.DataFrame(per_event).to_csv(OUT / 'V4_S03_truth_probabilities.csv', index=False)
        fig, ax = plt.subplots(figsize=(16, 4.8))
        centers = (section.start_s + section.end_s) / 120
        ax.plot(centers, section.p_line, color='#1679a7', linewidth=1.6, label='Probabilidad de lineout guardada')
        for i, t in enumerate(truth):
            ax.axvspan(t['start'] / 60, t['end'] / 60, color='#64748b', alpha=0.2, label='Lineout real' if i == 0 else None)
        for threshold, color, style in [(0.7, '#be3445', '--'), (0.65, '#ac741d', ':'), (0.55, '#50863f', '--')]:
            ax.axhline(threshold, color=color, linestyle=style, linewidth=1.2, label=f'Umbral {threshold:.2f}')
        ax.set(xlim=(0, 20), ylim=(0, 1), xlabel='Minutos de la sección S03', ylabel='Probabilidad de lineout')
        ax.set_xticks(np.arange(0, 21, 1))
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Auditoría V4 · las omisiones ya están en las probabilidades históricas', fontsize=17, x=0.05, ha='left')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.25, top=0.87)
        fig.savefig(OUT / 'V4_S03_probability_audit.png', dpi=170, facecolor='white')
        plt.close(fig)
pd.DataFrame(rows).to_csv(OUT / 'independent_baseline_by_match.csv', index=False)
older = []
for n in range(1, 7):
    ref = pd.read_csv(_resolve_model_artifact(BASE / f'PV-M1-H2-D2-PV0{n}-S42/attempt_001/predictions.csv'))
    for recipe in ['B-NA', 'H1-D2', 'H2-D2']:
        seed = 0 if recipe == 'B-NA' else 42
        raw = BASE / f'PV-M1-{recipe}-PV0{n}-S{seed}/attempt_001/predictions.csv'
        f = pd.read_csv(_resolve_model_artifact(raw))
        assert np.array_equal(f.block_id, ref.block_id) and np.array_equal(f.y_true, ref.y_true)
        f['audit_prediction'] = f.y_final
        res, *_ = evaluate(f, 'line')
        older.append(dict(match=f'V_{n}', recipe=recipe, scope='complete_match', **res))
        if n == 4:
            s = f[f.section == 'S03']
            res, *_ = evaluate(s, 'line')
            older.append(dict(match='V_4', recipe=recipe, scope='S03_20min', **res))
pd.DataFrame(older).to_csv(OUT / 'older_m1_comparison.csv', index=False)
result = dict(passed=True, independent_implementation=True, checks=checks, raw_probability_hashes=sources, explanation='V4 S03 belongs to the weakest held-out fold, not the VF all-development model. No degradation from changing weights occurred in this experiment.')
(OUT / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
report = ['AUDITORÍA DE LA TIMELINE V4 S03', '', 'La figura mostró V4 S03 (TRL B C L S J vs UNI, 11/04/2026), del pliegue PV04. No mostró VF (Jockey Club VT vs Alma Jr, 30/09/2023). El modelo de PV04 excluye V4 del entrenamiento CV; el modelo único de VF se entrenó sobre los seis partidos y CH. Son pesos y partidos distintos.', 'Se reprodujeron las decisiones desde los CSV originales, sin importar los auxiliares de métricas del experimento. Se reconstruyó además la fusión geométrica desde las probabilidades guardadas de dual y geometry. Se reprodujeron agrupación, anclas, emparejamientos y métricas históricas de ambas clases, y los intervalos que usa la timeline. Los hashes de las probabilidades coinciden con los de la primera evaluación.', 'La presentación anterior no dejó suficientemente claro que se había elegido el partido de peor rendimiento. Las cifras no son una caída nueva causada por este experimento. Tampoco representan el rendimiento típico de M1 en todos los partidos.', '', pd.DataFrame(rows).to_string(index=False), '', 'V4 S03: barrido independiente. Los umbrales de scrum y las probabilidades se conservan.', pd.DataFrame(summary).to_string(index=False), '', 'Resultados anteriores de VF: F1 temporal medio ambas clases por sección 0.683928 con umbrales y 0.743841 con argmax. El F1 temporal de lineout es 0.658516 en ambos casos. Eso no es comparable directamente con V4 S03 ni con la media de seis pliegues 0.600817.', 'Otra diferencia concreta: M1-B congelado (pose32, argmax) puede recuperar muchas más jugadas que H2-D2 en V4, pero con muchas más falsas alarmas. H2-D2 fue elegido por el promedio de F1 temporal de ambas clases entre los seis pliegues, no por ser el que detecta más lineouts en cada partido. M1-H1-D2 también supera a H2-D2 para lineout en V4, aunque tiene menor promedio global.', pd.DataFrame(older).to_string(index=False), 'No se reextrajeron características ni se ejecutaron redes. No se vuelve a establecer aquí la corrección semántica de las etiquetas mirando el video: se verifica su correspondencia con las fuentes manuales originales.']
(OUT / 'reporte_auditoria.txt').write_text('\n'.join(report), encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(summary).to_string(index=False))
print('Independent checks passed:', len(checks))
print(pd.DataFrame(older)[pd.DataFrame(older).match == 'V_4'].to_string(index=False))
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
specs = [('M1-B anterior · argmax', 'PV-M1-B-NA-PV04-S0', None), ('M1-H2-D2 · umbral 0.70', 'PV-M1-H2-D2-PV04-S42', 0.7), ('M1-H2-D2 · umbral 0.40', 'PV-M1-H2-D2-PV04-S42', 0.4)]
fig, ax = plt.subplots(figsize=(19, 7.7))
targets = None
plotrows = []
for title, rid, threshold in specs:
    f = pd.read_csv(_resolve_model_artifact(BASE / rid / 'attempt_001/predictions.csv'))
    f = f[f.section == 'S03'].copy()
    f['audit_prediction'] = f.y_final if threshold is None else decision(f, threshold, 0.4)
    res, t, p, pairs = evaluate(f, 'line')
    targets = t
    plotrows.append((title, res, t, p, pairs))
colors = dict(real='#64748b', tp='#1679a7', fp='#dc7a18', miss='#be3445')
ax.axhline(3, color='#f0f3f6', linewidth=27, zorder=0)
for t in targets:
    ax.broken_barh([(t['start'] / 60, (t['end'] - t['start']) / 60)], (2.82, 0.36), facecolors=colors['real'], zorder=3)
for k, (title, res, t, p, pairs) in enumerate(plotrows):
    y = 2 - k
    matched_t = {x[0] for x in pairs}
    matched_p = {x[1] for x in pairs}
    ax.axhline(y, color='#f0f3f6', linewidth=27, zorder=0)
    for j, e in enumerate(p):
        ax.broken_barh([(e['start'] / 60, (e['end'] - e['start']) / 60)], (y - 0.18, 0.36), facecolors=colors['tp' if j in matched_p else 'fp'], zorder=3)
    for i, e in enumerate(t):
        if i not in matched_t:
            ax.scatter([e['anchor'] / 60], [y], marker='x', color=colors['miss'], s=75, linewidths=2, zorder=5)
    ax.text(20.3, y, f"{res['tp']} detectados\n{res['fp']} falsas alarmas\n{res['fn']} omitidos", va='center', fontsize=12)
ax.set_yticks([3, 2, 1, 0], ['Lineouts reales'] + [s[0] for s in specs], fontsize=14)
ax.tick_params(axis='y', length=0, pad=15)
ax.set(xlim=(0, 20), ylim=(-0.55, 3.55), xlabel='Tiempo desde el comienzo de la sección (min:s)')
ax.set_xticks(np.arange(21), [f'{i:02d}:00' for i in range(21)], fontsize=11)
ax.xaxis.grid(True, color='#dce3ea', linewidth=0.8)
ax.spines[['left', 'right', 'top']].set_visible(False)
ax.spines['bottom'].set_color('#dce3ea')
fig.suptitle('Mismo video: M1 anterior frente a M1-H2-D2', x=0.035, y=0.97, ha='left', fontsize=23)
fig.text(0.035, 0.905, 'V4 · S03 · 20 minutos · mismas etiquetas y tolerancia de 35 s · predicciones históricas verificadas', fontsize=14, color=colors['real'])
fig.legend(handles=[Patch(facecolor=colors['real'], label='Lineout real'), Patch(facecolor=colors['tp'], label='Detectado'), Patch(facecolor=colors['fp'], label='Falsa alarma'), Line2D([], [], marker='x', linestyle='None', color=colors['miss'], label='Omitido', markersize=9)], ncol=4, loc='lower center', bbox_to_anchor=(0.54, 0.065), frameon=False, fontsize=13)
fig.text(0.035, 0.02, 'El umbral 0.40 es exploratorio. M1-B y M1-H2-D2 tienen pesos y arquitectura interna distintos; no representan un cambio de umbral solamente.', fontsize=12, color=colors['real'])
fig.subplots_adjust(left=0.24, right=0.88, top=0.84, bottom=0.22)
fig.savefig(OUT / 'V4_S03_old_vs_current_20min.png', dpi=180, facecolor='white')
plt.close(fig)
