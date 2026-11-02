"""Compare argmax with an absolute probability gate for lineout on fixed M1-H2-D2 outputs."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, hashlib, time
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
LABELS = np.array(['line', 'other', 'scrum'])
GRID = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
BASE = ROOT / 'data/final_model_comparison_20260827/outputs/runs'
SOURCES = {}

def track(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()
    return path

def readjson(path):
    return json.loads(_resolve_model_artifact(track(path)).read_text(encoding='utf-8'))
config = dict(reference='argmax over line,other,scrum, first index wins exact ties', model='M1-H2-D2', probability_sources='same held-out fold and attempt_001 for each match, unchanged weights', thresholds=GRID, threshold_source='absolute exploratory grid; argmax has no scalar threshold', scrum_priority='if argmax is scrum, keep scrum; otherwise assign line iff p_line >= t, else other', line_priority='assign line iff p_line >= t, otherwise retain argmax scrum if present, else other', other='The other probability no longer blocks lineout from being accepted by its absolute threshold.', rejection='If argmax is line but p_line < t, a pure threshold rule rejects it as other. This can occur for t > 1/3.', invariants='same probabilities, manual labels, intervals, grouping, 35 s tolerance; no training or feature extraction', matching='original maximum temporal-credit Hungarian one-to-one assignment, true midpoint / predicted peak-block midpoint', aggregation='equal weight per complete match; counts also summed', validation='Exploratory: same-fold development probabilities were not persisted; no independently validated new threshold', distinction='This replaces argmax with an absolute lineout gate; it is not the original normalized target-score threshold rule.')
(HERE / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
prior = readjson(ROOT / 'data/m1_line_threshold_sweep_20261022/validation.json')
original_inventory = readjson(ROOT / 'data/m1_line_threshold_sweep_20261022/inventory.json')
track(Path(m.__file__))
rows = []
derived = []
truth_changes = []
candidates = []
inventory = []
plot_frames = {}
checks = []
start = time.perf_counter()
for i in range(1, 7):
    match = f'V_{i}'
    rid = f'PV-M1-H2-D2-PV0{i}-S42'
    folder = BASE / rid / 'attempt_001'
    f = pd.read_csv(_resolve_model_artifact(track(folder / 'predictions.csv'))).rename(columns={'test_group': 'group_id'})
    meta = readjson(folder / 'run.json')
    assert SOURCES[str(folder / 'predictions.csv')] == prior['source_hashes'][str(folder / 'predictions.csv')]
    assert f.checkpoint_hash.eq(meta['checkpoint_hash']).all()
    assert match == meta['test_group'] and match not in meta['train_groups'] and (match != meta['development_group'])
    inv = next((s for s in original_inventory if s['scope'] == 'PV_heldout' and s['match'] == match))
    for path, sha in inv['weight_hashes'].items():
        assert SOURCES.setdefault(str(path), hashlib.sha256(_resolve_model_artifact(Path(path)).read_bytes()).hexdigest()) == sha
    inventory.append(inv)
    probs = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
    argmax = probs.argmax(1)
    assert np.array_equal(LABELS[argmax], f.y_argmax)
    hours = f.duration.sum() / 3600
    baseline = f.copy()
    baseline['y_final'] = LABELS[argmax]
    baselines = {}
    truths = {}
    for c in ['line', 'scrum']:
        truths[c] = m.events_from_blocks(f, c, prediction_column='y_true', probability_column='p_' + c)
        be = m.events_from_blocks(baseline, c, prediction_column='y_final', probability_column='p_' + c)
        bm = m.match_events(truths[c], be)
        baselines[c] = (be, bm)
    rules = [('argmax', None)] + [(rule, t) for rule in ['scrum_priority', 'line_priority'] for t in GRID]
    for rule, t in rules:
        if rule == 'argmax':
            y = argmax.copy()
        else:
            y = np.full(len(f), 1, dtype=int)
            y[argmax == 2] = 2
            accept = probs[:, 0] >= t
            if rule == 'scrum_priority':
                accept &= argmax != 2
            y[accept] = 0
        if rule == 'scrum_priority':
            assert np.array_equal(y == 2, argmax == 2)
        d = f.copy()
        d['y_final'] = LABELS[y]
        if match == 'V_4':
            plot_frames[rule, t] = d[d.section == 'S03'].copy()
        derived.append(d.assign(policy=rule, line_threshold=t)[['run_id', 'block_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'duration', 'y_true', 'p_line', 'p_other', 'p_scrum', 'y_argmax', 'y_final', 'policy', 'line_threshold', 'checkpoint_hash']])
        tm = m.temporal_metrics(d)
        row = dict(match=match, policy=rule, line_threshold=t, hours=hours, temporal_f1_macro=tm['weighted_f1_macro'], scrum_to_line_blocks=int(((argmax == 2) & (y == 0)).sum()), other_to_line_blocks=int(((argmax == 1) & (y == 0)).sum()), argmax_line_rejected_blocks=int(((argmax == 0) & (y != 0)).sum()), line_blocks=int((y == 0).sum()))
        for c in ['line', 'scrum']:
            truth = truths[c]
            pred = m.events_from_blocks(d, c, prediction_column='y_final', probability_column='p_' + c)
            pairs = m.match_events(truth, pred)
            mt = {a for a, b, dist, credit in pairs}
            mp = {b for a, b, dist, credit in pairs}
            basep, basepairs = baselines[c]
            bmt = {a for a, b, dist, credit in basepairs}
            rec = mt - bmt
            lost = bmt - mt
            fp = len(pred) - len(pairs)
            fn = len(truth) - len(pairs)
            tp = len(pairs)
            far = fp - tm[c]['duplicates']
            vals = dict(real=len(truth), tp=tp, fp=fp, fn=fn, precision=tp / len(pred) if pred else 0, recall=tp / len(truth) if truth else 0, f1=2 * tp / (len(truth) + len(pred)) if len(truth) + len(pred) else 1, temporal_f1=tm[c]['weighted_f1'], weighted_tp=sum((z[3] for z in pairs)), zero_credit_pairs=sum((z[3] == 0 for z in pairs)), fp_per_hour=fp / hours, far_fp=far, far_fp_per_hour=far / hours, duplicates=tm[c]['duplicates'], candidates=len(pred), candidate_seconds=sum((e.end - e.start for e in pred)), recovered=len(rec), lost=len(lost), fusions=tm[c]['fusions'], fragmentations=tm[c]['fragmentations'])
            row.update({c + '_' + k: v for k, v in vals.items()})
            for tid in sorted(rec | lost):
                e = truth[tid]
                truth_changes.append(dict(match=match, policy=rule, line_threshold=t, label=c, truth_id=tid, effect='recovered' if tid in rec else 'lost', section=e.section, start_s=e.start, end_s=e.end, anchor_s=e.anchor))
            for j, e in enumerate(pred):
                pair = next((z for z in pairs if z[1] == j), None)
                nearby = any((v.section == e.section and abs(v.anchor - e.anchor) <= 35 for v in truth))
                overlap_base = [b for b, v in enumerate(basep) if v.section == e.section and min(v.end, e.end) > max(v.start, e.start)]
                candidates.append(dict(match=match, policy=rule, line_threshold=t, label=c, event_id=j, section=e.section, start_s=e.start, end_s=e.end, anchor_s=e.anchor, status='TP' if pair else 'duplicate_FP' if nearby else 'far_FP', truth_id=pair[0] if pair else None, distance_s=pair[2] if pair else None, weighted_credit=pair[3] if pair else 0, baseline_overlapping_ids=','.join(map(str, overlap_base)), merged_baseline_candidates=len(overlap_base) > 1))
        row['all_fp_per_hour'] = row['line_fp_per_hour'] + row['scrum_fp_per_hour']
        row['all_far_fp_per_hour'] = row['line_far_fp_per_hour'] + row['scrum_far_fp_per_hour']
        rows.append(row)
    checks.append(dict(match=match, argmax_reproduced=True, probabilities_equal_previous=True, checkpoint_hash=meta['checkpoint_hash']))
    print('Evaluated', match, flush=True)
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / 'metrics_by_match.csv', index=False)
pd.concat(derived, ignore_index=True).to_csv(OUT / 'derived_predictions.csv.gz', index=False, compression='gzip')
pd.DataFrame(truth_changes).to_csv(OUT / 'recovered_lost_vs_argmax.csv', index=False)
pd.DataFrame(candidates).to_csv(OUT / 'candidate_events.csv.gz', index=False, compression='gzip')
aggregate = []
for (rule, t), g in metrics.groupby(['policy', 'line_threshold'], dropna=False, sort=False):
    r = dict(policy=rule, line_threshold=None if pd.isna(t) else t, matches=len(g))
    for col in g.select_dtypes(include='number').columns:
        if col != 'line_threshold':
            r['mean_' + col] = float(g[col].mean())
    for c in ['line', 'scrum']:
        for col in ['real', 'tp', 'fp', 'fn', 'candidates', 'far_fp', 'recovered', 'lost', 'fusions', 'fragmentations', 'zero_credit_pairs']:
            r['sum_' + c + '_' + col] = int(g[c + '_' + col].sum())
    for col in ['scrum_to_line_blocks', 'other_to_line_blocks', 'argmax_line_rejected_blocks']:
        r['sum_' + col] = int(g[col].sum())
    aggregate.append(r)
agg = pd.DataFrame(aggregate)
agg.to_csv(OUT / 'aggregate_equal_match_weight.csv', index=False)
old = pd.read_csv(_resolve_model_artifact(ROOT / 'data/m1_line_threshold_sweep_20261022/results/metrics_by_match.csv'))
for match, g in metrics[metrics.policy == 'argmax'].groupby('match'):
    previous = old[(old.scope == 'PV_heldout') & (old.match == match) & (old.policy == 'argmax')].iloc[0]
    assert np.isclose(g.iloc[0].temporal_f1_macro, previous.temporal_f1_macro)
assert all((hashlib.sha256(_resolve_model_artifact(Path(p)).read_bytes()).hexdigest() == sha for p, sha in SOURCES.items()))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': '#202a35'})
selected = [('argmax', None), ('scrum_priority', 0.4), ('scrum_priority', 0.3), ('scrum_priority', 0.2), ('scrum_priority', 0.1)]
truth = m.events_from_blocks(plot_frames[selected[0]], 'line', prediction_column='y_true', probability_column='p_line')
fig, ax = plt.subplots(figsize=(17, 9))
colors = dict(real='#64748b', tp='#1679a7', fp='#dc7a18', miss='#be3445')
for e in truth:
    ax.axvspan(e.start / 60, e.end / 60, color=colors['real'], alpha=0.045, zorder=0)
    ax.broken_barh([(e.start / 60, (e.end - e.start) / 60)], (4.82, 0.36), facecolors=colors['real'])
labels = ['Lineouts reales']
sectionrows = []
for k, key in enumerate(selected):
    d = plot_frames[key]
    pred = m.events_from_blocks(d, 'line', prediction_column='y_final', probability_column='p_line')
    pairs = m.match_events(truth, pred)
    mt = {z[0] for z in pairs}
    mp = {z[1] for z in pairs}
    y = 4 - k
    ax.axhline(y, color='#eef2f6', linewidth=25, zorder=0)
    for j, e in enumerate(pred):
        ax.broken_barh([(e.start / 60, (e.end - e.start) / 60)], (y - 0.18, 0.36), facecolors=colors['tp' if j in mp else 'fp'])
    for i, e in enumerate(truth):
        if i not in mt:
            ax.scatter([e.anchor / 60], [y], marker='x', s=60, color=colors['miss'], linewidth=1.8, zorder=5)
    labels.append('Argmax · referencia' if key[0] == 'argmax' else f'p(lineout) ≥ {key[1]:.2f}')
    ax.text(20.3, y, f'{len(pairs)}/10 detectados\n{len(pred) - len(pairs)} falsas alarmas', va='center', fontsize=12)
    sectionrows.append(dict(policy=key[0], line_threshold=key[1], real=len(truth), tp=len(pairs), fp=len(pred) - len(pairs), fn=len(truth) - len(pairs)))
pd.DataFrame(sectionrows).to_csv(OUT / 'V4_S03_timeline_counts.csv', index=False)
ax.set_yticks([5, 4, 3, 2, 1, 0], labels, fontsize=13)
ax.tick_params(axis='y', length=0, pad=12)
ax.set(xlim=(0, 20), ylim=(-0.55, 5.5), xlabel='Tiempo desde el comienzo de la sección (min:s)')
ax.set_xticks(np.arange(0, 21, 2), [f'{i:02d}:00' for i in range(0, 21, 2)])
ax.xaxis.grid(True, color='#dce3ea')
ax.spines[['left', 'right', 'top']].set_visible(False)
fig.suptitle('M1-H2-D2: argmax frente a umbrales absolutos de lineout', x=0.035, y=0.97, ha='left', fontsize=21)
fig.text(0.035, 0.92, 'Mismas probabilidades · V4 / S03 · 20 minutos · los scrum de argmax se conservan', fontsize=13, color=colors['real'])
fig.legend(handles=[Patch(facecolor=colors['real'], label='Lineout real'), Patch(facecolor=colors['tp'], label='Detectado'), Patch(facecolor=colors['fp'], label='Falsa alarma'), Line2D([], [], marker='x', color=colors['miss'], linestyle='None', label='Omitido')], ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.52, 0.065), fontsize=12)
fig.text(0.035, 0.02, 'El umbral acepta lineout aunque other tenga mayor probabilidad. Evaluación por evento con tolerancia de 35 s. Barrido exploratorio.', fontsize=11, color=colors['real'])
fig.subplots_adjust(left=0.2, right=0.84, top=0.87, bottom=0.17)
fig.savefig(FIG / 'argmax_vs_line_thresholds_V4_S03_20min.png', dpi=170, facecolor='white')
plt.close(fig)
lines = ['M1-H2-D2: ARGMAX FRENTE A UMBRAL ABSOLUTO DE LINEOUT', '', 'Este experimento evalúa sustituir la decisión por argmax por una aceptación mediante p(lineout) >= t, aun si other tiene una probabilidad mayor. La referencia es argmax, no los umbrales históricos de cada pliegue.', 'Se usan los mismos pesos y probabilidades de prueba de M1-H2-D2 de los seis partidos completos. Etiquetas y procedencia fueron verificadas contra las sesiones manuales en el experimento anterior; se comprueba que sus hashes no cambiaron.', 'Regla principal, scrum_priority: si argmax es scrum se conserva scrum. En los demás bloques se acepta lineout si su probabilidad supera o iguala el umbral, y se asigna other en caso contrario. Así se elimina la competencia con other y se mantiene idéntica toda la secuencia de scrum.', 'Alternativa, line_priority: lineout gana siempre que alcanza el umbral. Si no lo alcanza, se conserva scrum cuando era argmax, y se usa other en los demás casos. Aquí un lineout aceptado puede desplazar a scrum. La cantidad de desplazamientos se exporta y no se mezcla con la regla principal.', 'Argmax no tiene un umbral numérico original. Para t > 1/3 un lineout que ganaba argmax con confianza baja puede ser rechazado. Para t <= 0.30 todos los lineouts de argmax quedan admitidos por el umbral, pero los eventos pueden unirse y su número o emparejamiento puede cambiar.', 'Se conservan agrupación por bloques consecutivos dentro de cada sección, ancla real central, ancla predicha en el centro del primer bloque de probabilidad máxima, asignación húngara uno a uno y tolerancia de 35 s. TP/FP/FN son enteros; el F1 temporal pondera las parejas por distancia. Todas las medias dan el mismo peso a cada partido.', 'Los umbrales se comparan usando etiquetas de prueba y son exploratorios. Faltan probabilidades de validación de los mismos pesos por pliegue para recomendar un nuevo umbral con estimación fuera de muestra. No hubo inferencia, entrenamiento, cambios en pesos o aplicación.', '', agg[['policy', 'line_threshold', 'sum_line_tp', 'sum_line_fp', 'sum_line_fn', 'mean_line_precision', 'mean_line_recall', 'mean_line_temporal_f1', 'mean_scrum_temporal_f1', 'mean_temporal_f1_macro', 'mean_line_fp_per_hour', 'mean_line_far_fp_per_hour', 'mean_all_fp_per_hour', 'mean_all_far_fp_per_hour', 'sum_line_recovered', 'sum_line_lost', 'sum_scrum_to_line_blocks']].to_string(index=False)]
(HERE / 'reporte.txt').write_text('\n'.join(lines), encoding='utf-8')
(HERE / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
(HERE / 'verification.json').write_text(json.dumps(dict(checks=checks, all_sources_unchanged=True, source_hashes=SOURCES, baseline_matches_prior_argmax=True, seconds=time.perf_counter() - start, prediction_rows=sum((len(x) for x in derived))), indent=2), encoding='utf-8')
print(agg[['policy', 'line_threshold', 'sum_line_tp', 'sum_line_fp', 'sum_line_fn', 'mean_line_precision', 'mean_line_recall', 'mean_line_temporal_f1', 'mean_scrum_temporal_f1', 'mean_temporal_f1_macro', 'sum_line_recovered', 'sum_line_lost', 'sum_scrum_to_line_blocks']].to_string(index=False))
