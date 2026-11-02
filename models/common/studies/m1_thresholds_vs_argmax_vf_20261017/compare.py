"""Paired decision-rule comparison on unchanged saved VF probabilities."""
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
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from models.common.comparison import metrics as m
from models.common.comparison.protocol import LABELS, LABEL_TO_ID
HERE = experiment_directory(__file__)
CAMPAIGNS = {'S01': ROOT / 'data/m1_ensemble_vs_all_development_20261012', **{f'S0{i}': ROOT / f'data/m1_ensemble_vs_single_vf_s0{i}_20261013' for i in (2, 3, 4)}}
HUMAN = {s: folder / 'inputs/human_blocks.csv' for s, folder in CAMPAIGNS.items()}
HUMAN['S01'] = ROOT / 'data/active_learning_dataset/sessions/20260830_112524_0b5e58/train_manifest.csv'
sources = [Path(m.__file__), _resolve_model_artifact(ROOT / 'models/common/comparison/protocol.py')]
for folder in CAMPAIGNS.values():
    for key in ('M1-single', 'M1-ensemble'):
        sources += [folder / 'results' / f'{key}_quality_blocks.csv', folder / 'results' / f'{key}_quality.json', folder / 'results' / f'{key}_startup.json']
sources += list(HUMAN.values())
assert all((_resolve_model_artifact(p).exists() for p in sources)), [str(p) for p in sources if not _resolve_model_artifact(p).exists()]

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()
before = {str(p): sha(p) for p in sources}
start = time.perf_counter()
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
rows = []
details = []
changes = []
paired = []
frames = {}
all_frame = []
reference_checks = []

def evaluate(frame, prediction):
    f = frame.copy()
    f['y_final'] = np.asarray(LABELS)[prediction]
    truth = f['y_true'].map(LABEL_TO_ID).to_numpy()
    wrong = prediction != truth
    temporal = m.temporal_metrics(f)
    precision, recall, f1, support = precision_recall_fscore_support(truth, prediction, labels=np.arange(3), zero_division=0)
    block = {'n': len(f), 'errors': int(wrong.sum()), 'error_rate': float(wrong.mean()), 'wrong_seconds': float(f.loc[wrong, 'duration'].sum()), 'error_time_fraction': float(f.loc[wrong, 'duration'].sum() / f['duration'].sum()), 'f1_macro': float(f1_score(truth, prediction, labels=np.arange(3), average='macro', zero_division=0)), 'confusion_matrix': confusion_matrix(truth, prediction, labels=np.arange(3)).tolist(), 'per_class': {label: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i, label in enumerate(LABELS)}}
    for label in ('line', 'scrum'):
        truth_events = m.events_from_blocks(f, label, prediction_column='y_true', probability_column='p_' + label)
        pred_events = m.events_from_blocks(f, label, prediction_column='y_final', probability_column='p_' + label)
        matches = m.match_events(truth_events, pred_events)
        tm = temporal[label]
        tm['false_negative_events'] = len(truth_events) - len(matches)
        tm['unmatched_predicted_events'] = len(pred_events) - len(matches)
        tm['weighted_true_positive_credit'] = sum((x[3] for x in matches))
        tm['zero_credit_matches'] = sum((x[3] == 0 for x in matches))
        tm['far_false_positive_count'] = tm['unmatched_predicted_events'] - tm['duplicates']
    return (f, block, temporal)
for segment, folder in CAMPAIGNS.items():
    human = pd.read_csv(_resolve_model_artifact(HUMAN[segment])).rename(columns={'dataset_id': 'block_id', 'candidate_index': 'block_index', 'correct_label': 'y_true', 'start_time': 'start_s', 'end_time': 'end_s'})
    for key in ('M1-single', 'M1-ensemble'):
        f = pd.read_csv(_resolve_model_artifact(folder / 'results' / f'{key}_quality_blocks.csv'))
        assert len(f) == 172 and f['section'].eq(segment).all()
        assert np.isclose(f['duration'].sum(), 1201.2) and np.isclose(f.iloc[-1]['duration'], 4.2)
        assert len(human) == len(f)
        for col in ('block_id', 'block_index', 'y_true', 'start_s', 'end_s'):
            assert col in human
            if col in ('start_s', 'end_s'):
                assert np.allclose(f[col], human[col], atol=1e-07), (segment, key, col)
            else:
                assert np.array_equal(f[col].to_numpy(), human[col].to_numpy()), (segment, key, col)
        frame_key = (key, segment)
        p = f[['p_line', 'p_other', 'p_scrum']].to_numpy()
        assert np.isfinite(p).all() and np.all(p >= 0) and np.allclose(p.sum(axis=1), 1, atol=1e-06)
        startup = json.loads(_resolve_model_artifact(folder / 'results' / f'{key}_startup.json').read_text(encoding='utf-8'))
        thresholds = startup['thresholds']
        assert np.isclose(thresholds['line'], 0.5) and np.isclose(thresholds['scrum'], 0.45)
        pred = {'thresholds': m.apply_class_thresholds(p, line=thresholds['line'], scrum=thresholds['scrum']), 'argmax': m.normalize_probabilities(p).argmax(axis=1)}
        assert np.array_equal(np.asarray(LABELS)[pred['thresholds']], f['y_final'].to_numpy()), (segment, key, 'decision reproduction')
        old = json.loads(_resolve_model_artifact(folder / 'results' / f'{key}_quality.json').read_text(encoding='utf-8'))
        current = {}
        for rule, labels in pred.items():
            out, block, temporal = evaluate(f, labels)
            out.to_csv(OUT / f'{segment}_{key}_{rule}_blocks.csv', index=False)
            frames[key, segment, rule] = out
            all_frame.append(out.assign(model=key, segment=segment, rule=rule))
            if rule == 'thresholds':
                assert abs(temporal['weighted_f1_macro'] - old['temporal']['weighted_f1_macro']) < 1e-06
                assert all((abs(temporal[label]['weighted_f1'] - old['temporal'][label]['weighted_f1']) < 1e-06 for label in ('line', 'scrum')))
                reference_checks.append({'model': key, 'segment': segment, 'saved_decisions_reproduced': True, 'temporal_f1_reproduced': True})
            row = {'model': key, 'segment': segment, 'rule': rule, 'blocks': len(f), 'block_errors': block['errors'], 'block_error_pct': 100 * block['error_rate'], 'wrong_seconds': block['wrong_seconds'], 'block_f1_macro': block['f1_macro'], 'temporal_f1_macro': temporal['weighted_f1_macro'], 'temporal_f1_line': temporal['line']['weighted_f1'], 'temporal_f1_scrum': temporal['scrum']['weighted_f1'], 'event_omissions': sum((temporal[label]['false_negative_events'] for label in ('line', 'scrum'))), 'unmatched_predictions': sum((temporal[label]['unmatched_predicted_events'] for label in ('line', 'scrum'))), 'far_false_positives': sum((temporal[label]['far_false_positive_count'] for label in ('line', 'scrum'))), 'duplicates': sum((temporal[label]['duplicates'] for label in ('line', 'scrum'))), 'fragmentations': sum((temporal[label]['fragmentations'] for label in ('line', 'scrum'))), 'fusions': sum((temporal[label]['fusions'] for label in ('line', 'scrum'))), 'zero_credit_matches': sum((temporal[label]['zero_credit_matches'] for label in ('line', 'scrum'))), 'predicted_minutes': sum((temporal[label]['predicted_minutes'] for label in ('line', 'scrum')))}
            rows.append(row)
            current[rule] = row
            details.append({'model': key, 'segment': segment, 'rule': rule, 'thresholds': {'line': thresholds['line'], 'scrum': thresholds['scrum']} if rule == 'thresholds' else None, 'block': block, 'temporal': temporal})
        truth = f['y_true'].map(LABEL_TO_ID).to_numpy()
        a, t = (pred['argmax'], pred['thresholds'])
        changed = a != t
        fixed = (a != truth) & (t == truth)
        broken = (a == truth) & (t != truth)
        paired.append({'model': key, 'segment': segment, 'changed_blocks': int(changed.sum()), 'argmax_errors_corrected_by_thresholds': int(fixed.sum()), 'argmax_correct_broken_by_thresholds': int(broken.sum()), 'changed_but_both_wrong': int((changed & ~fixed & ~broken).sum()), 'delta_temporal_f1_thresholds_minus_argmax': current['thresholds']['temporal_f1_macro'] - current['argmax']['temporal_f1_macro']})
        for i in np.flatnonzero(changed):
            changes.append({'model': key, 'segment': segment, **f.iloc[i][['block_id', 'block_index', 'start_s', 'end_s', 'duration', 'y_true', 'p_line', 'p_other', 'p_scrum']].to_dict(), 'argmax': LABELS[a[i]], 'thresholds': LABELS[t[i]], 'effect': 'corrects_argmax' if fixed[i] else 'breaks_argmax_correct' if broken[i] else 'both_wrong'})
        if key == 'M1-ensemble':
            other = frames['M1-single', segment, 'thresholds']
            assert np.array_equal(other['y_true'], f['y_true'])
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'metrics_by_segment.csv', index=False)
pd.DataFrame(paired).to_csv(OUT / 'paired_changes.csv', index=False)
pd.DataFrame(changes).to_csv(OUT / 'changed_blocks.csv', index=False)
pd.concat(all_frame, ignore_index=True).to_csv(OUT / 'all_decisions.csv', index=False)
aggregates = []
for key in ('M1-single', 'M1-ensemble'):
    for rule in ('thresholds', 'argmax'):
        selected = summary[(summary.model == key) & (summary.rule == rule)]
        aggregates.append({'model': key, 'rule': rule, 'segments': 4, 'blocks': int(selected.blocks.sum()), 'block_errors': int(selected.block_errors.sum()), 'block_error_pct': 100 * float(selected.block_errors.sum() / selected.blocks.sum()), 'mean_segment_temporal_f1': float(selected.temporal_f1_macro.mean()), 'mean_segment_line_f1': float(selected.temporal_f1_line.mean()), 'mean_segment_scrum_f1': float(selected.temporal_f1_scrum.mean()), **{col: int(selected[col].sum()) for col in ('event_omissions', 'unmatched_predictions', 'far_false_positives', 'duplicates', 'fragmentations', 'fusions', 'zero_credit_matches')}, 'wrong_seconds': float(selected.wrong_seconds.sum()), 'predicted_minutes': float(selected.predicted_minutes.sum())})
pd.DataFrame(aggregates).to_csv(OUT / 'aggregate_summary.csv', index=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))
colors = {'argmax': '#7c8b9c', 'thresholds': '#126d83'}
for ri, key in enumerate(('M1-single', 'M1-ensemble')):
    for ci, (column, title) in enumerate((('temporal_f1_macro', 'F1 temporal medio · mayor es mejor'), ('block_error_pct', 'Bloques mal clasificados (%) · menor es mejor'))):
        ax = axes[ri, ci]
        y = np.arange(4)
        for shift, rule, label in ((-0.17, 'argmax', 'Argmax'), (0.17, 'thresholds', 'Umbrales 0.50 / 0.45')):
            v = summary[(summary.model == key) & (summary.rule == rule)].set_index('segment').loc[list(CAMPAIGNS), column].to_numpy()
            ax.barh(y + shift, v, height=0.29, color=colors[rule], label=label)
            for yy, x in zip(y + shift, v):
                ax.text(x + 0.01 if ci == 0 else x + 0.22, yy, f'{x:.3f}' if ci == 0 else f'{x:.1f}', ha='left', va='center', fontsize=10)
        ax.set_yticks(y, list(CAMPAIGNS))
        ax.invert_yaxis()
        ax.set_xlim(0, 1 if ci == 0 else max(20, summary.block_error_pct.max() * 1.28))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, alpha=0.2)
        ax.set_title(('Modelo único' if ri == 0 else 'Ensemble de seis folds') + '\n' + title, loc='left', fontsize=12, pad=12)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper right', bbox_to_anchor=(0.97, 0.98), frameon=False, fontsize=10)
fig.suptitle('M1-H2-D2: umbrales frente a argmax', fontsize=21, x=0.07, ha='left', y=0.97)
fig.text(0.07, 0.905, 'Mismas probabilidades y etiquetas humanas. Cuatro segmentos de VF; 172 bloques por segmento.', fontsize=11)
fig.text(0.07, 0.035, 'F1: promedio de lineout y scrum por segmento, protocolo temporal de la tesina.\nResultados descriptivos sobre un mismo video; no se ajustaron modelos ni umbrales con VF.', fontsize=10, color='#444444')
fig.subplots_adjust(left=0.07, right=0.96, top=0.84, bottom=0.13, hspace=0.55, wspace=0.28)
fig.savefig(FIG / 'comparacion_umbrales_argmax.png', dpi=180, facecolor='white')
plt.close(fig)
after = {str(p): sha(p) for p in sources}
assert before == after
elapsed = time.perf_counter() - start
result = {'scope': 'paired re-evaluation on existing probabilities; no new video inference or training', 'labels_order': LABELS, 'threshold_policy': 'exact original apply_class_thresholds: line .50, scrum .45; compare p_line/t_line with p_scrum/t_scrum; otherwise other', 'argmax_policy': 'maximum probability over line/other/scrum, numpy first index on ties', 'reference_checks': reference_checks, 'metrics': details, 'aggregates': aggregates, 'paired_changes': paired, 'source_hashes_before': before, 'source_hashes_after': after, 'sources_unchanged': before == after, 'analysis_seconds': elapsed, 'independent_test_claim_allowed': False, 'segment_boundaries': 'Each section evaluated separately exactly as before. No claim of reconstructed full-match events.', 'temporal_metric': 'one-to-one global assignment, max(0,1-distance/35), true midpoint / predicted peak-block midpoint', 'human_grid': '172 blocks per segment: 171 x 7 s plus 4.2 s', 'timing': 'Only reevaluation and export time; original video inference timings unchanged.'}
(OUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
report = ['M1-H2-D2: UMBRALES FRENTE A ARGMAX', '17 de octubre de 2026', '', 'Comparación pareada sobre las probabilidades ya guardadas del modelo único y del ensemble, en VF_S01–VF_S04. No se volvió a ejecutar ResNet, YOLO ni las redes, ni se entrenó o ajustó ningún componente.', '', 'RESULTADO OBSERVADO', 'En el modelo único, argmax eleva la media de F1 temporal por segmento de 0.683928 a 0.743841, con 58 errores por bloque en lugar de 59 sobre 688 bloques. Las omisiones se mantienen en 7 y las predicciones sin pareja bajan de 14 a 11 (FP lejanos de 7 a 5 y duplicados de 7 a 6).', 'En el ensemble, argmax eleva esa media de 0.622114 a 0.636995, aunque comete 65 errores por bloque frente a 64. Las omisiones se mantienen en 16 y las predicciones sin pareja bajan de 5 a 4. Son resultados descriptivos de estos cuatro segmentos, no una selección nueva de parámetros.', 'Los umbrales cambian cinco decisiones del modelo único: corrigen dos errores de argmax y convierten tres aciertos en errores. En el ensemble cambian tres decisiones: corrigen dos y perjudican una. El F1 de lineout no cambia en ningún segmento; la diferencia temporal se concentra en scrum.', '', 'REGLAS DE DECISIÓN', 'Umbrales: lineout 0.50 y scrum 0.45, usando exactamente la función original de la tesina. Se comparan las puntuaciones relativas p_line/0.50 y p_scrum/0.45; una clase objetivo debe alcanzar su umbral y ganar esa comparación. En otro caso se asigna otro.', 'Argmax: se elige la mayor de las tres probabilidades, incluida otro. No es equivalente a aplicar un umbral uniforme de 0.50. Puede admitir una clase objetivo con probabilidad inferior al umbral si es la más alta de las tres. Con estos umbrales, también puede ocurrir que scrum supere 0.45 y desplace a otro aunque otro tenga una probabilidad ligeramente mayor.', '', 'MÉTRICAS POR SEGMENTO', 'Modelo | Segmento | Regla | F1 temporal | Errores/bloques | Error % | Omisiones | Predicciones sin pareja | FP lejanos | Duplicados']
for r in rows:
    report.append(f"{r['model']} | {r['segment']} | {r['rule']} | {r['temporal_f1_macro']:.6f} | {r['block_errors']}/{r['blocks']} | {r['block_error_pct']:.3f} | {r['event_omissions']} | {r['unmatched_predictions']} | {r['far_false_positives']} | {r['duplicates']}")
report += ['', 'RESUMEN DE LOS CUATRO SEGMENTOS', 'Modelo | Regla | Media de F1 por segmento | Errores/688 | Error % | Omisiones | Predicciones sin pareja | FP lejanos | Duplicados']
for r in aggregates:
    report.append(f"{r['model']} | {r['rule']} | {r['mean_segment_temporal_f1']:.6f} | {r['block_errors']}/688 | {r['block_error_pct']:.3f} | {r['event_omissions']} | {r['unmatched_predictions']} | {r['far_false_positives']} | {r['duplicates']}")
report += ['', 'CAMBIOS AL APLICAR UMBRALES SOBRE ARGMAX', 'Modelo | Segmento | Bloques que cambian | Errores corregidos | Aciertos que se vuelven errores | Cambio de F1 temporal']
for r in paired:
    report.append(f"{r['model']} | {r['segment']} | {r['changed_blocks']} | {r['argmax_errors_corrected_by_thresholds']} | {r['argmax_correct_broken_by_thresholds']} | {r['delta_temporal_f1_thresholds_minus_argmax']:+.6f}")
report += ['', 'LECTURA Y ALCANCE', 'Ejemplo concreto del modelo único en S01, 10:16–10:23: p_otro=0.53896236 y p_scrum=0.45151427. Argmax elige otro, que coincide con la etiqueta humana. La regla de umbrales elige scrum porque supera 0.45 y gana la comparación relativa entre clases objetivo, aunque otro tenga mayor probabilidad.', 'En S03, el modelo único conserva el mismo número total de errores por bloque (20), pero los errores cambian de lugar y clase. Con argmax propone el único scrum real y obtiene F1 de scrum 0.90. Al aplicar umbrales agrega un scrum falso en 15:31–15:38 y su F1 de scrum baja a 0.60. El F1 temporal medio del segmento pasa de 0.734211 con argmax a 0.584211 con umbrales. La corrección de un bloque de lineout en otra zona no cambia sus eventos de lineout.', 'El error por bloque cuenta decisiones distintas de la clase humana. El F1 temporal reagrupa las decisiones y evalúa eventos: una decisión que cambia puede eliminar una falsa alarma, pero también cortar, unir o borrar una jugada. Por eso ambos criterios pueden moverse en direcciones distintas.', 'Omisiones significa eventos reales sin pareja bajo el emparejamiento original. Predicciones sin pareja incluye FP lejanos y duplicados cercanos. Se informa por separado cualquier pareja con crédito cero en los resultados detallados. FP lejanos no reúne todas las predicciones erróneas.', 'El resumen muestra la media de los cuatro F1 por segmento y suma recuentos de esas evaluaciones separadas. No equivale al F1 del partido completo reconstruido, porque aquí se mantienen los límites de sección del análisis original.', 'Los cuatro segmentos pertenecen al mismo video. La independencia de todos los clips históricos respecto de VF no está completamente verificada: estos resultados siguen siendo descriptivos. Esta comparación no se usó para modificar los umbrales ni elegir nuevos parámetros.', 'Se reprodujeron las decisiones con umbrales y los F1 temporales originales de las ocho corridas antes de compararlos con argmax. Se conservaron los hashes de probabilidades, etiquetas y código.', f'Tiempo del análisis y generación de archivos: {elapsed:.2f} s. No es tiempo de inferencia de video.', '', 'ARCHIVOS', 'results/metrics_by_segment.csv: las 16 combinaciones de modelo, segmento y regla.', 'results/aggregate_summary.csv: resumen por modelo y regla.', 'results/paired_changes.csv: correcciones y regresiones por segmento.', 'results/changed_blocks.csv: bloques que cambian con referencia humana y probabilidades.', 'results/all_decisions.csv: todas las decisiones, sin modificar las fuentes.', 'results/comparison.json: métricas por clase, matrices de confusión, verificación y hashes.', 'figures/comparacion_umbrales_argmax.png: comparación visual.']
(HERE / 'reporte.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(pd.DataFrame(aggregates).to_json(orient='records', indent=2))
print(pd.DataFrame(paired).to_json(orient='records', indent=2))
print(f'ANALYSIS_SECONDS={elapsed:.3f}')
