"""Show unchanged M0-H2-D0/D2 outputs using argmax on the six held-out matches."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, hashlib, time
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score
from models.common.comparison import metrics as m
from models.common.comparison.protocol import LABELS, LABEL_TO_ID
BASE = ROOT / 'data/final_model_comparison_20260827/outputs'
OUT = HERE / 'results'
FIG = HERE / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
paths = [Path(m.__file__), BASE / 'completed_fold_results.csv', BASE / 'comparison_summary.csv']
selected = {}
registry = pd.read_csv(_resolve_model_artifact(BASE / 'completed_fold_results.csv'))
for regime in ('D0', 'D2'):
    for index in range(1, 7):
        run = f'PV-M0-H2-{regime}-PV0{index}-S42'
        ref = registry[registry.run_id.eq(run)]
        assert len(ref) == 1
        candidates = []
        for folder in _model_artifact_glob(BASE / 'runs' / run, 'attempt_*', recursive=False):
            if not all((_resolve_model_artifact(folder / name).exists() for name in ('run.json', 'metrics.json', 'predictions.csv'))):
                continue
            meta = json.loads(_resolve_model_artifact(folder / 'run.json').read_text(encoding='utf-8'))
            saved = json.loads(_resolve_model_artifact(folder / 'metrics.json').read_text(encoding='utf-8'))
            if meta['smoke_limit'] is None and abs(saved['temporal']['weighted_f1_macro'] - ref.iloc[0].f1w_macro) < 1e-09:
                candidates.append(folder)
        assert len(candidates) == 1, (run, candidates)
        selected[regime, index] = candidates[0]
        paths += [candidates[0] / name for name in ('run.json', 'metrics.json', 'predictions.csv')]

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()
before = {str(p): sha(p) for p in paths}
start = time.perf_counter()
rows = []
details = []
frames = {}
checks = []
for (regime, index), folder in selected.items():
    original = pd.read_csv(_resolve_model_artifact(folder / 'predictions.csv'))
    meta = json.loads(_resolve_model_artifact(folder / 'run.json').read_text(encoding='utf-8'))
    old = json.loads(_resolve_model_artifact(folder / 'metrics.json').read_text(encoding='utf-8'))
    f = original.copy()
    f['group_id'] = f['test_group']
    assert f['test_group'].eq(f'V_{index}').all()
    assert len(f) == meta['n_test']
    p = f[['p_line', 'p_other', 'p_scrum']].to_numpy()
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-07)
    argmax = np.asarray(LABELS)[m.normalize_probabilities(p).argmax(axis=1)]
    assert np.array_equal(argmax, f['y_argmax'])
    thresholds = meta['thresholds']
    reproduced = np.asarray(LABELS)[m.apply_class_thresholds(p, line=thresholds['line'], scrum=thresholds['scrum'])]
    assert np.array_equal(reproduced, original['y_final'])
    if regime == 'D2':
        other = frames['D0', index, 'argmax']
        assert np.array_equal(f.block_id, other.block_id) and np.array_equal(f.y_true, other.y_true)
        assert np.allclose(f[['start_s', 'end_s']], other[['start_s', 'end_s']])
    for rule, values in (('thresholds', original.y_final.to_numpy()), ('argmax', argmax)):
        out = f.copy()
        out['y_final'] = values
        frames[regime, index, rule] = out
        temporal = m.temporal_metrics(out)
        truth = out.y_true.map(LABEL_TO_ID).to_numpy()
        pred = out.y_final.map(LABEL_TO_ID).to_numpy()
        errors = int((truth != pred).sum())
        if rule == 'thresholds':
            assert abs(temporal['weighted_f1_macro'] - old['temporal']['weighted_f1_macro']) < 1e-09
            checks.append({'run_id': meta['run_id'], 'original_threshold_decisions_reproduced': True, 'original_temporal_f1_reproduced': True, 'stored_argmax_reproduced': True})
        row = {'model': f'M0-H2-{regime}', 'regime': regime, 'match': f'V{index}', 'fold': meta['fold'], 'rule': rule, 'blocks': len(out), 'block_errors': errors, 'block_error_pct': 100 * errors / len(out), 'block_f1_macro': float(f1_score(truth, pred, labels=np.arange(3), average='macro', zero_division=0)), 'temporal_f1_macro': temporal['weighted_f1_macro'], 'temporal_f1_line': temporal['line']['weighted_f1'], 'temporal_f1_scrum': temporal['scrum']['weighted_f1'], 'far_false_positives_per_hour': sum((temporal[label]['far_false_positives_per_hour'] for label in ('line', 'scrum'))), 'event_omissions': sum((temporal[label]['true_events'] - temporal[label]['matched_events'] for label in ('line', 'scrum'))), 'unmatched_predictions': sum((temporal[label]['predicted_events'] - temporal[label]['matched_events'] for label in ('line', 'scrum'))), 'predicted_line_events': temporal['line']['predicted_events'], 'predicted_scrum_events': temporal['scrum']['predicted_events'], 'duplicates': sum((temporal[label]['duplicates'] for label in ('line', 'scrum'))), 'line_threshold': thresholds['line'] if rule == 'thresholds' else None, 'scrum_threshold': thresholds['scrum'] if rule == 'thresholds' else None}
        rows.append(row)
        details.append({'run_id': meta['run_id'], 'rule': rule, 'block_confusion_matrix': confusion_matrix(truth, pred, labels=np.arange(3)).tolist(), 'labels_order': LABELS, 'temporal': temporal})
        out.to_csv(OUT / f'M0-H2-{regime}_V{index}_{rule}_blocks.csv', index=False)
df = pd.DataFrame(rows)
df.to_csv(OUT / 'metrics_by_match.csv', index=False)
summary = []
for regime in ('D0', 'D2'):
    for rule in ('thresholds', 'argmax'):
        sub = df[df.regime.eq(regime) & df.rule.eq(rule)]
        assert len(sub) == 6 and sub.blocks.sum() == 4714
        summary.append({'model': f'M0-H2-{regime}', 'rule': rule, 'matches': 6, 'mean_temporal_f1': float(sub.temporal_f1_macro.mean()), 'mean_temporal_f1_line': float(sub.temporal_f1_line.mean()), 'mean_temporal_f1_scrum': float(sub.temporal_f1_scrum.mean()), 'mean_far_false_positives_per_hour': float(sub.far_false_positives_per_hour.mean()), 'block_errors': int(sub.block_errors.sum()), 'blocks': int(sub.blocks.sum()), 'pooled_block_error_pct': 100 * float(sub.block_errors.sum() / sub.blocks.sum()), 'event_omissions_sum': int(sub.event_omissions.sum()), 'unmatched_predictions_sum': int(sub.unmatched_predictions.sum())})
pd.DataFrame(summary).to_csv(OUT / 'six_match_summary.csv', index=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
colors = {'line': '#248a5a', 'scrum': '#4269b2', 'other': '#e8ebef'}
fig, ax = plt.subplots(figsize=(15, 5.4))
plot_rows = [frames['D0', 2, 'argmax'].query("section == 'S01'").copy(), frames['D0', 2, 'argmax'].query("section == 'S01'").copy(), frames['D2', 2, 'argmax'].query("section == 'S01'").copy()]
plot_rows[0]['y_final'] = plot_rows[0]['y_true']
names = ['Real', 'M0-H2-D0 · argmax', 'M0-H2-D2 · argmax']
window_metrics = []
for y, (name, f) in enumerate(zip(names, plot_rows)):
    assert len(f) == 171 and np.isclose(f.end_s.max(), 1197)
    for label, color in colors.items():
        intervals = [(r.start_s / 60, (r.end_s - r.start_s) / 60) for r in f.itertuples() if r.y_final == label]
        ax.broken_barh(intervals, (y - 0.27, 0.54), facecolors=color, edgecolors='none')
    ax.broken_barh([(1197 / 60, 3 / 60)], (y - 0.27, 0.54), facecolors='#bbbbbb', edgecolors='#777777', hatch='///')
    if y > 0:
        window_metrics.append({'model': names[y].split(' · ')[0], 'window': 'V2 S01: 00:00–19:57 annotated; 19:57–20:00 unannotated', 'temporal': m.temporal_metrics(f)})
ax.set_yticks(range(3), names)
ax.set_ylim(2.65, -0.7)
ax.set_xlim(0, 20)
ax.set_xticks(np.arange(0, 21, 2), [f'{i:02d}:00' for i in range(0, 21, 2)])
ax.set_xlabel('Tiempo dentro del primer segmento de V2 (min:s)', labelpad=12)
ax.grid(axis='x', alpha=0.15)
ax.set_axisbelow(True)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0, pad=13)
fig.suptitle('V2 · Real y predicciones sin umbrales', x=0.04, y=0.97, ha='left', fontsize=21)
fig.text(0.04, 0.875, 'Mismas probabilidades guardadas. Decisión: clase de mayor probabilidad entre lineout, otro y scrum.', fontsize=11)
fig.legend(handles=[Patch(color=colors['line'], label='Lineout'), Patch(color=colors['scrum'], label='Scrum'), Patch(color=colors['other'], label='Otro'), Patch(facecolor='#bbbbbb', hatch='///', label='Sin anotación')], loc='lower center', bbox_to_anchor=(0.54, 0.065), ncol=4, frameon=False, fontsize=11)
fig.text(0.04, 0.025, 'PV02: V2 reservado para prueba y V3 para desarrollo. Vista de 20 minutos; últimos 3 s sin etiquetas.', fontsize=10, color='#444444')
fig.subplots_adjust(left=0.19, right=0.97, top=0.79, bottom=0.29)
fig.savefig(FIG / 'v2_20min_real_m0_d0_d2_argmax.png', dpi=180, facecolor='white')
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 5.4))
positions = np.arange(6)
for offset, regime, color in ((-0.18, 'D0', '#a66a32'), (0.18, 'D2', '#126d83')):
    sub = df[df.regime.eq(regime) & df.rule.eq('argmax')].sort_values('match')
    bars = ax.bar(positions + offset, sub.temporal_f1_macro, width=0.32, color=color, label=f'M0-H2-{regime} · argmax')
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
ax.set_xticks(positions, [f'V{i}' for i in range(1, 7)])
ax.set_ylim(0, 1)
ax.set_ylabel('F1 temporal medio de lineout y scrum')
ax.set_axisbelow(True)
ax.yaxis.grid(True, alpha=0.2)
ax.legend(frameon=False, loc='upper right')
ax.set_title('M0 sin umbrales · los seis partidos de prueba', loc='left', fontsize=17, pad=18)
fig.text(0.09, 0.025, 'Cada partido se evalúa con su fold reservado. No se volvió a entrenar ni ejecutar inferencia de video.', fontsize=9, color='#444444')
fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.14)
fig.savefig(FIG / 'f1_argmax_seis_partidos.png', dpi=180, facecolor='white')
plt.close(fig)
after = {str(p): sha(p) for p in paths}
assert after == before
elapsed = time.perf_counter() - start
result = {'date': '2026-10-21', 'all_checks_pass': True, 'original_runs_reproduced': checks, 'sources_unchanged': after == before, 'source_hashes_before': before, 'source_hashes_after': after, 'summary': summary, 'metrics': details, 'V2_first_20min': window_metrics, 'analysis_seconds': elapsed, 'policy': 'argmax over line, other, scrum; models and probabilities unchanged', 'no_new_video_inference': True, 'no_new_training': True, 'no_threshold_tuning': True, 'scope': 'Panel V held-out per-match evaluation; does not refer to VF ensemble or single-all-development M1.'}
(OUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
report = ['M0-H2-D0 Y M0-H2-D2 SIN UMBRALES', '21 de octubre de 2026', '', 'Se recalculó argmax sobre las probabilidades guardadas de las doce corridas originales: dos modelos por seis folds. Se mantienen modelos, entradas y etiquetas. No se repitió extracción ni inferencia de video.', 'En D0 se entrena con clips iniciales; en D2 con clips y los cuatro partidos de entrenamiento de cada fold. Se evalúan los seis partidos reservados de Panel V. No son las pruebas VF del ensemble M1.', 'La regla sin umbrales selecciona la mayor probabilidad de las tres clases. Se conserva el F1 temporal de la tesina, con referencias, agrupación por sección y emparejamiento uno a uno sin cambios.', '', 'RESULTADOS SIN UMBRALES POR PARTIDO', 'Modelo | Partido | F1 temporal | F1 lineout | F1 scrum | FP lejanos/h | Error por bloque %']
for r in rows:
    if r['rule'] == 'argmax':
        report.append(f"{r['model']} | {r['match']} | {r['temporal_f1_macro']:.6f} | {r['temporal_f1_line']:.6f} | {r['temporal_f1_scrum']:.6f} | {r['far_false_positives_per_hour']:.3f} | {r['block_error_pct']:.3f}")
report += ['', 'PROMEDIO DE LOS SEIS PARTIDOS Y COMPARACIÓN CON UMBRALES', 'Modelo | Decisión | F1 temporal medio | F1 lineout medio | F1 scrum medio | FP lejanos/h medio | Errores/4714']
for r in summary:
    report.append(f"{r['model']} | {r['rule']} | {r['mean_temporal_f1']:.6f} | {r['mean_temporal_f1_line']:.6f} | {r['mean_temporal_f1_scrum']:.6f} | {r['mean_far_false_positives_per_hour']:.3f} | {r['block_errors']}/4714")
report += ['', 'V2 COMPLETO: COMPARACIÓN DIRECTA', 'Modelo | Decisión | F1 temporal | FP lejanos/h | Eventos de lineout predichos | Eventos de scrum predichos | Omisiones']
for r in rows:
    if r['match'] == 'V2':
        report.append(f"{r['model']} | {r['rule']} | {r['temporal_f1_macro']:.6f} | {r['far_false_positives_per_hour']:.3f} | {r['predicted_line_events']} | {r['predicted_scrum_events']} | {r['event_omissions']}")
report += ['', 'ALCANCE', 'Las métricas por partido corresponden a todo su material anotado. La imagen de veinte minutos sólo muestra S01 de V2, de 00:00 a 20:00. Hay 171 bloques etiquetados que llegan a 19:57; los últimos tres segundos se señalan como sin anotación.', 'Los promedios temporales y de FP lejanos/h dan igual peso a los seis partidos, como en la tesina. Los errores por bloque se suman sobre los 4714 bloques. No son el mismo tipo de agregación.', 'Se verificó que argmax reproduce y_argmax guardado y que los umbrales originales reproducen y_final y el F1 temporal registrado en las doce corridas. Las anotaciones y los bloques de D0 y D2 coinciden en cada partido. Las fuentes mantienen sus hashes.', 'La trazabilidad histórica de los clips e inicializaciones sigue siendo una limitación del estudio. Este análisis conserva el protocolo existente y no afirma una nueva prueba externa.', f'Tiempo de reevaluación y generación de resultados: {elapsed:.2f} segundos. No es una medición de inferencia de video.', '', 'ARCHIVOS', 'results/metrics_by_match.csv: métricas de las 24 combinaciones de modelo, partido y regla.', 'results/six_match_summary.csv: promedio de los seis partidos.', 'results/comparison.json: resultados detallados por clase, verificaciones y hashes.', 'results/*_blocks.csv: decisiones por bloque, sin modificar los archivos originales.', 'figures/v2_20min_real_m0_d0_d2_argmax.png: Real y ambas configuraciones sin umbrales.', 'figures/f1_argmax_seis_partidos.png: F1 por partido sin umbrales.']
(HERE / 'reporte.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(pd.DataFrame(summary).to_json(orient='records', indent=2))
print(df[df.match.eq('V2')].to_json(orient='records', indent=2))
print(f'ANALYSIS_SECONDS={elapsed:.3f}')
