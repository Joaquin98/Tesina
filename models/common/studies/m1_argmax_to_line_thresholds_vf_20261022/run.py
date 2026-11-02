"""Apply the already specified lineout gates to the additional VF test match."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, hashlib, sys, time
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
SOURCES = {}

def track(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()
    return path

def readjson(path):
    return json.loads(_resolve_model_artifact(track(path)).read_text(encoding='utf-8'))
VFROOT = ROOT / 'data/m1_ensemble_vs_all_development_20261012'
manifest = readjson(VFROOT / 'inputs/M1-single_manifest.json')
independence = readjson(VFROOT / 'inputs/independence_audit.json')
prior = readjson(ROOT / 'data/m1_line_threshold_sweep_20261022/validation.json')
fixed_config = readjson(ROOT / 'data/m1_argmax_to_line_thresholds_20261022/config.json')
assert manifest['source_configuration'] == 'ALL-DEVELOPMENT-M1-H2-D2-S42'
assert fixed_config['reference'].startswith('argmax')
weights = {}
for key in ['dual_model', 'geometry_rf']:
    p = track(ROOT / manifest['folds'][0][key])
    weights[str(p)] = SOURCES[str(p)]
sources = []
frames = []
checks = []
track(Path(m.__file__))
for i in range(1, 5):
    section = f'S0{i}'
    campaign = VFROOT if i == 1 else ROOT / f'data/m1_ensemble_vs_single_vf_s0{i}_20261013'
    probability = track(campaign / 'results/M1-single_quality_blocks.csv')
    assert SOURCES[str(probability)] == prior['source_hashes'][str(probability)]
    f = pd.read_csv(_resolve_model_artifact(probability))
    assert len(f) == 172 and f.section.eq(section).all() and f.group_id.eq('VF').all()
    assert np.isclose(f.duration.sum(), 1201.2)
    startup = readjson(campaign / 'results/M1-single_startup.json')
    historical = readjson(campaign / 'results/M1-single_quality.json')
    for path, sha in weights.items():
        assert startup['hashes_before'][path] == sha
    humanpath = ROOT / 'data/active_learning_dataset/sessions/20260830_112524_0b5e58/train_manifest.csv' if i == 1 else campaign / 'inputs/human_blocks.csv'
    human = pd.read_csv(_resolve_model_artifact(track(humanpath))).rename(columns={'dataset_id': 'block_id', 'candidate_index': 'block_index', 'correct_label': 'y_true', 'start_time': 'start_s', 'end_time': 'end_s'})
    for c in ['block_id', 'block_index', 'y_true']:
        assert np.array_equal(f[c], human[c])
    for c in ['start_s', 'end_s']:
        assert np.allclose(f[c], human[c])
    p = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
    original = m.apply_class_thresholds(p, line=startup['thresholds']['line'], scrum=startup['thresholds']['scrum'])
    assert np.array_equal(np.array(['line', 'other', 'scrum'])[original], f.y_final)
    assert np.isclose(m.temporal_metrics(f)['weighted_f1_macro'], historical['temporal']['weighted_f1_macro'])
    sources.append(dict(section=section, probability_path=str(probability), label_path=str(humanpath), blocks=len(f), seconds=float(f.duration.sum())))
    checks.append(dict(section=section, same_single_model_weights=True, probabilities_unchanged=True, labels_verified=True, historical_baseline_reproduced=True))
    frames.append(f)
full = pd.concat(frames, ignore_index=True)
assert len(full) == 688
POLICIES = [('argmax', None), ('scrum_priority', 0.45), ('scrum_priority', 0.35), ('line_priority', 0.45), ('line_priority', 0.35)]
config = dict(match='VF: Jockey Club VT vs Alma Juniors, 30/09/2023', source_configuration=manifest['source_configuration'], model='single H2-D2 refit on CH and all V1-V6; same frozen single-model weights for every rule', weights=weights, coverage='S01-S04 only: 4804.8 s / 80.08 min, separately grouped as in the original protocol', missing='VF S05: no complete M1-single probability sequence and verified manual labels found; small legacy graph scene files are not compatible coverage', policies=POLICIES, thresholds_fixed_before_this_run=[0.45, 0.35], primary='Keep argmax scrum; elsewhere line iff p_line >= threshold, otherwise other. Other does not compete.', alternative='Line has priority even when scrum was argmax. Evaluated separately.', invariants='same probabilities, weights, labels, sections, grouping, anchors, one-to-one Hungarian credit matching, tolerance 35 s', aggregation='primary match count/credit pooled over S01-S04 with sections kept separate; equal-section means also reported and explicitly named', interpretation='Diagnostic only. VF absent from declared CV training groups, but legacy historical parent-video provenance is incomplete; prior VF label analyses exist.')
(HERE / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

def evaluate(frame):
    tm = m.temporal_metrics(frame)
    hours = float(frame.duration.sum() / 3600)
    row = dict(hours=hours, temporal_f1_macro=tm['weighted_f1_macro'])
    details = {}
    for c in ['line', 'scrum']:
        truth = m.events_from_blocks(frame, c, prediction_column='y_true', probability_column='p_' + c)
        pred = m.events_from_blocks(frame, c, prediction_column='y_final', probability_column='p_' + c)
        pairs = m.match_events(truth, pred)
        tp = len(pairs)
        fp = len(pred) - tp
        fn = len(truth) - tp
        vals = dict(real=len(truth), tp=tp, fp=fp, fn=fn, precision=tp / len(pred) if pred else 0, recall=tp / len(truth) if truth else 0, f1=2 * tp / (len(truth) + len(pred)) if len(truth) + len(pred) else 1, temporal_f1=tm[c]['weighted_f1'], weighted_tp=sum((z[3] for z in pairs)), fp_per_hour=fp / hours, far_fp=fp - tm[c]['duplicates'], far_fp_per_hour=(fp - tm[c]['duplicates']) / hours, candidates=len(pred), candidate_seconds=sum((e.end - e.start for e in pred)), duplicates=tm[c]['duplicates'], fusions=tm[c]['fusions'], fragmentations=tm[c]['fragmentations'], zero_credit_pairs=sum((z[3] == 0 for z in pairs)))
        row.update({c + '_' + k: v for k, v in vals.items()})
        details[c] = (truth, pred, pairs)
    row['all_fp_per_hour'] = row['line_fp_per_hour'] + row['scrum_fp_per_hour']
    row['all_far_fp_per_hour'] = row['line_far_fp_per_hour'] + row['scrum_far_fp_per_hour']
    return (row, details)
p = m.normalize_probabilities(full[['p_line', 'p_other', 'p_scrum']].to_numpy())
argmax = p.argmax(1)
labels = np.array(['line', 'other', 'scrum'])
baseline = full.copy()
baseline['y_final'] = labels[argmax]
base_row, base_detail = evaluate(baseline)
match_rows = []
section_rows = []
truth_changes = []
candidate_rows = []
derived = []
plots = {}
for rule, t in POLICIES:
    if rule == 'argmax':
        y = argmax.copy()
    else:
        y = np.full(len(full), 1, dtype=int)
        y[argmax == 2] = 2
        accept = p[:, 0] >= t
        if rule == 'scrum_priority':
            accept &= argmax != 2
        y[accept] = 0
    if rule == 'scrum_priority':
        assert np.array_equal(y == 2, argmax == 2)
    d = full.copy()
    d['y_argmax'] = labels[argmax]
    d['y_final'] = labels[y]
    derived.append(d.assign(policy=rule, line_threshold=t))
    plots[rule, t] = d
    row, detail = evaluate(d)
    row.update(policy=rule, line_threshold=t, scrum_to_line_blocks=int(((argmax == 2) & (y == 0)).sum()))
    for c in ['line', 'scrum']:
        truth, pred, pairs = detail[c]
        mt = {z[0] for z in pairs}
        mp = {z[1] for z in pairs}
        bt = {z[0] for z in base_detail[c][2]}
        row[c + '_recovered'] = len(mt - bt)
        row[c + '_lost'] = len(bt - mt)
        for i in sorted(mt ^ bt):
            e = truth[i]
            truth_changes.append(dict(policy=rule, line_threshold=t, label=c, effect='recovered' if i in mt else 'lost', truth_id=i, section=e.section, start_s=e.start, end_s=e.end, anchor_s=e.anchor))
        for j, e in enumerate(pred):
            pair = next((z for z in pairs if z[1] == j), None)
            nearby = any((a.section == e.section and abs(a.anchor - e.anchor) <= 35 for a in truth))
            candidate_rows.append(dict(policy=rule, line_threshold=t, label=c, event_id=j, section=e.section, start_s=e.start, end_s=e.end, anchor_s=e.anchor, status='TP' if pair else 'duplicate_FP' if nearby else 'far_FP', truth_id=pair[0] if pair else None, distance_s=pair[2] if pair else None, weighted_credit=pair[3] if pair else 0))
    for sec, s in d.groupby('section', sort=False):
        r, _ = evaluate(s)
        section_rows.append(dict(section=sec, policy=rule, line_threshold=t, **r))
    sections = [r for r in section_rows if r['policy'] == rule and (t is None and r['line_threshold'] is None or r['line_threshold'] == t)]
    assert sum((r['line_tp'] for r in sections)) == row['line_tp']
    assert sum((r['line_fp'] for r in sections)) == row['line_fp']
    row['mean_section_temporal_f1_macro'] = float(np.mean([r['temporal_f1_macro'] for r in sections]))
    row['mean_section_line_temporal_f1'] = float(np.mean([r['line_temporal_f1'] for r in sections]))
    match_rows.append(row)
match_metrics = pd.DataFrame(match_rows)
section_metrics = pd.DataFrame(section_rows)
match_metrics.to_csv(OUT / 'match_metrics.csv', index=False)
section_metrics.to_csv(OUT / 'metrics_by_section.csv', index=False)
pd.concat(derived, ignore_index=True).to_csv(OUT / 'derived_predictions.csv.gz', index=False, compression='gzip')
pd.DataFrame(truth_changes).to_csv(OUT / 'recovered_lost_events.csv', index=False)
pd.DataFrame(candidate_rows).to_csv(OUT / 'candidate_events.csv', index=False)
assert np.isclose(match_rows[0]['mean_section_temporal_f1_macro'], 0.7438414375092007, atol=1e-06)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': '#202a35'})
fig, axes = plt.subplots(4, 1, figsize=(16, 11))
colors = dict(real='#64748b', tp='#1679a7', fp='#dc7a18', miss='#be3445')
for ax, (sec, _) in zip(axes, full.groupby('section', sort=False)):
    f = baseline[baseline.section == sec]
    truth = m.events_from_blocks(f, 'line', prediction_column='y_true', probability_column='p_line')
    for e in truth:
        ax.broken_barh([(e.start / 60, (e.end - e.start) / 60)], (2.83, 0.34), facecolors=colors['real'])
    rowlabels = ['Reales']
    for i, key in enumerate([('argmax', None), ('scrum_priority', 0.45), ('scrum_priority', 0.35)]):
        f = plots[key]
        f = f[f.section == sec]
        pred = m.events_from_blocks(f, 'line', prediction_column='y_final', probability_column='p_line')
        pairs = m.match_events(truth, pred)
        mt = {z[0] for z in pairs}
        mp = {z[1] for z in pairs}
        y = 2 - i
        ax.axhline(y, color='#f1f4f7', linewidth=17, zorder=0)
        for j, e in enumerate(pred):
            ax.broken_barh([(e.start / 60, (e.end - e.start) / 60)], (y - 0.17, 0.34), facecolors=colors['tp' if j in mp else 'fp'])
        for j, e in enumerate(truth):
            if j not in mt:
                ax.scatter([e.anchor / 60], [y], color=colors['miss'], marker='x', s=40, linewidth=1.5)
        ax.text(20.25, y, f'{len(pairs)}/{len(truth)} detectados · {len(pred) - len(pairs)} FP', va='center', fontsize=10)
        rowlabels.append('Argmax' if key[0] == 'argmax' else f'Umbral {key[1]:.2f}')
    ax.set_yticks([3, 2, 1, 0], rowlabels, fontsize=11)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.set(xlim=(0, 20.02), ylim=(-0.45, 3.5))
    ax.set_xticks(np.arange(0, 21, 2), [f'{v:02d}:00' for v in range(0, 21, 2)], fontsize=10)
    ax.xaxis.grid(True, color='#dce3ea')
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.set_title(f'{sec} · aproximadamente 20 minutos', loc='left', fontsize=12, pad=8)
axes[-1].set_xlabel('Tiempo local de cada sección (min:s)', labelpad=10)
fig.suptitle('Partido de test VF · argmax frente a umbrales bajos de lineout', x=0.035, y=0.98, ha='left', fontsize=21)
fig.text(0.035, 0.942, 'Jockey Club VT vs Alma Juniors · M1 único H2-D2 · mismas probabilidades · scrum conservado', fontsize=12, color=colors['real'])
fig.legend(handles=[Patch(facecolor=colors['real'], label='Lineout real'), Patch(facecolor=colors['tp'], label='Detectado'), Patch(facecolor=colors['fp'], label='Falsa alarma'), Line2D([], [], marker='x', color=colors['miss'], linestyle='None', label='Omitido')], ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.52, 0.033), fontsize=11)
fig.subplots_adjust(left=0.13, right=0.84, top=0.89, bottom=0.13, hspace=0.65)
fig.savefig(FIG / 'VF_80min_argmax_vs_thresholds.png', dpi=170, facecolor='white')
plt.close(fig)
assert all((hashlib.sha256(_resolve_model_artifact(Path(path)).read_bytes()).hexdigest() == sha for path, sha in SOURCES.items()))
verification = dict(checks=checks, sources_unchanged=True, source_hashes=SOURCES, coverage_seconds=float(full.duration.sum()), covered_sections=4, blocks=len(full), manual_lineout_events=base_row['line_real'], independence=independence)
(HERE / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
(HERE / 'inventory.json').write_text(json.dumps(dict(model=manifest, sources=sources, independence=independence), indent=2), encoding='utf-8')
columns = ['policy', 'line_threshold', 'line_real', 'line_tp', 'line_fp', 'line_fn', 'line_precision', 'line_recall', 'line_temporal_f1', 'scrum_temporal_f1', 'temporal_f1_macro', 'line_recovered', 'line_lost', 'line_fp_per_hour', 'line_far_fp_per_hour', 'all_fp_per_hour', 'all_far_fp_per_hour', 'scrum_to_line_blocks']
report = ['VF: ARGMAX FRENTE A UMBRALES ABSOLUTOS DE LINEOUT', '', 'Se aplicaron sin volver a elegirlos los umbrales 0.45 y 0.35 especificados antes de este análisis, sobre probabilidades existentes del modelo único ALL-M1-H2-D2-S42. Este es el refit de la configuración H2-D2 sobre CH y los seis partidos de desarrollo, no uno de los seis pesos de prueba ni un ensemble.', 'Regla principal: conservar scrum si es argmax. En los otros bloques aceptar lineout si p_line >= t, aunque other sea mayor, y other en caso contrario. La variante que da prioridad a lineout se informa por separado. Se conservan todos los parámetros de eventos y tolerancia 35 s.', 'Cobertura: S01-S04, 688 bloques y 4804.8 s (80.08 min). No se incorporó S05: no se encontró una secuencia completa de probabilidades del mismo M1 único con etiquetas humanas verificadas. Los resultados no cubren ese tramo final. Se mantienen los límites de sección del protocolo original.', 'Métricas del conjunto de cuatro secciones: conteos y crédito temporal sumados manteniendo separados sus eventos. Las medias con igual peso por sección se guardan explícitamente como mean_section_*. No son la media de seis partidos de desarrollo.', '', match_metrics[columns].to_string(index=False), '', 'Diagnóstico sobre el partido adicional: VF no aparece en los grupos CV de entrenamiento, pero la procedencia de los videos originales de CH está incompleta y VF ya fue analizado en trabajos anteriores. No se afirma independencia completa ni una confirmación final libre de selección. No se entrenó ni ejecutó inferencia.', 'Se verificaron hashes iguales en los cuatro tramos para dual.keras y geometry_rf.joblib, probabilidades sin cambios, etiquetas e intervalos iguales a las fuentes humanas, y reproducción de las métricas históricas.']
(HERE / 'reporte.txt').write_text('\n'.join(report), encoding='utf-8')
print(match_metrics[columns].to_string(index=False))
