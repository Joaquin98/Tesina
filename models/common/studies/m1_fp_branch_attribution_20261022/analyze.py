"""Event FP audit from saved branch probabilities. No inference or training.

Attribution is decision support at the fused event's confidence anchor under
the same policy. Whole-sequence single-branch counterfactuals are also matched
to truth; overlapping blocks associate their events with the original FP.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, html
sys.path.insert(0, str(source_directory(experiment_directory(__file__).parents[1])))
from models.common.studies.m1_argmax_grid_20261022.run import np, pd, m, decide, RUNS, LABELS, sha
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
records = []
block_records = []
checks = []
hashes = {}
reference = pd.read_csv(_resolve_model_artifact(HERE.parent / 'm1_arithmetic_vs_geometric_20261022/results/fixed_rules_by_fold.csv'))

def event_pack(frame, p, y, c):
    d = frame.copy()
    d['y_final'] = LABELS[y]
    for j, label in enumerate(LABELS):
        d['p_' + label] = p[:, j]
    truth = m.events_from_blocks(d, c, prediction_column='y_true', probability_column='p_' + c)
    events = m.events_from_blocks(d, c, prediction_column='y_final', probability_column='p_' + c)
    pairs = m.match_events(truth, events)
    return (truth, events, {j for _, j, _, _ in pairs})

def cf_status(original, events, matched):
    overlap = [j for j, e in enumerate(events) if (e.group, e.section) == (original.group, original.section) and set(e.blocks) & set(original.blocks)]
    fp = sum((j not in matched for j in overlap))
    tp = len(overlap) - fp
    status = 'Sin detección superpuesta' if not overlap else 'Sólo aciertos superpuestos' if not fp else 'Sólo FP superpuestos' if not tp else 'Aciertos y FP superpuestos'
    return (status, fp, tp)
for i in range(1, 7):
    fold = f'PV{i:02d}'
    folder = RUNS / f'PV-M1-H2-D2-{fold}-S42' / 'attempt_001'
    for filename in ['run.json', 'predictions.csv']:
        path = folder / filename
        hashes[str(path)] = sha(path)
    run = json.loads(_resolve_model_artifact(folder / 'run.json').read_text())
    f = pd.read_csv(_resolve_model_artifact(folder / 'predictions.csv'))
    f['group_id'] = f.test_group
    alpha = float(run['model_metadata']['geometry_alpha'])
    dual = m.normalize_probabilities(f[['p_dual_' + c for c in LABELS]].to_numpy())
    geom = m.normalize_probabilities(f[['p_geometry_' + c for c in LABELS]].to_numpy())
    stored = f[['p_' + c for c in LABELS]].to_numpy()
    arithmetic = m.normalize_probabilities((1 - alpha) * dual + alpha * geom)
    policies = [('argmax', 'argmax', 'argmax'), ('line_argmax_scrum_035', 'argmax', 0.35), ('line_050_scrum_035', 0.5, 0.35), ('original_fold_thresholds', run['thresholds']['line'], run['thresholds']['scrum'])]
    for policy, line, scrum in policies:
        yd = decide(dual, line, scrum)
        yg = decide(geom, line, scrum)
        for fusion, p in [('geometric', stored), ('arithmetic', arithmetic)]:
            p = m.normalize_probabilities(p)
            y = decide(p, line, scrum)
            ref = reference[(reference.fold == fold) & (reference.fusion == fusion) & (reference.policy == policy)].iloc[0]
            for c, k in [('line', 0), ('scrum', 2)]:
                truth, events, matched = event_pack(f, p, y, c)
                _, de, dm = event_pack(f, dual, yd, c)
                _, ge, gm = event_pack(f, geom, yg, c)
                fps = [j for j in range(len(events)) if j not in matched]
                assert len(fps) == int(ref[c + '_fp']), (fold, policy, fusion, c, len(fps), ref[c + '_fp'])
                checks.append(dict(fold=fold, policy=policy, fusion=fusion, event_class=c, fp=len(fps)))
                for j in fps:
                    e = events[j]
                    ids = f.index[(f.group_id == e.group) & (f.section == e.section) & f.block_index.isin(e.blocks)].to_numpy()
                    ids = ids[np.argsort(f.loc[ids, 'start_s'].to_numpy(), kind='stable')]
                    anchor = int(ids[np.argmax(p[ids, k])])
                    assert abs((f.loc[anchor, 'start_s'] + f.loc[anchor, 'end_s']) / 2 - e.anchor) < 1e-08
                    ds, gs = (yd[anchor] == k, yg[anchor] == k)
                    support = 'Ambas' if ds and gs else 'Sólo dual' if ds else 'Sólo geometría' if gs else 'Sólo combinación'
                    nearby = [t for t in truth if (t.group, t.section) == (e.group, e.section)]
                    nearest = min(nearby, key=lambda t: abs(t.anchor - e.anchor)) if nearby else None
                    distance = abs(nearest.anchor - e.anchor) if nearest else None
                    overlap = any((set(t.blocks) & set(e.blocks) for t in nearby))
                    fp_type = 'Duplicada / competencia de asignación' if distance is not None and distance <= 35 else 'Desplazada (>35 s), con solapamiento real' if overlap else 'Sin jugada de esa clase a <=35 s'
                    onlyd, dfp, dtp = cf_status(e, de, dm)
                    onlyg, gfp, gtp = cf_status(e, ge, gm)
                    ident = f'{fold}_{fusion}_{policy}_{c}_{j:03d}'
                    r = dict(fp_id=ident, fold=fold, test_group=e.group, section=e.section, event_class=c, fusion=fusion, policy=policy, line_rule=line, scrum_rule=scrum, alpha=alpha, start_s=e.start, end_s=e.end, anchor_s=e.anchor, anchor_block_id=f.loc[anchor, 'block_id'], anchor_block_index=int(f.loc[anchor, 'block_index']), truth_at_anchor=f.loc[anchor, 'y_true'], fp_type=fp_type, nearest_truth_anchor_s=nearest.anchor if nearest else None, nearest_truth_distance_s=distance, branch_support_at_anchor=support, dual_decision=LABELS[yd[anchor]], geometry_decision=LABELS[yg[anchor]], fused_decision=LABELS[y[anchor]], dual_argmax=LABELS[dual[anchor].argmax()], geometry_argmax=LABELS[geom[anchor].argmax()], fused_argmax=LABELS[p[anchor].argmax()], threshold_changes_anchor_decision=bool(p[anchor].argmax() != k), event_blocks=len(ids), dual_support_blocks=int((yd[ids] == k).sum()), geometry_support_blocks=int((yg[ids] == k).sum()), only_dual_event_status=onlyd, only_dual_overlapping_fp=dfp, only_dual_overlapping_tp=dtp, only_geometry_event_status=onlyg, only_geometry_overlapping_fp=gfp, only_geometry_overlapping_tp=gtp)
                    for name, probs in [('dual', dual), ('geometry', geom), ('fused', p)]:
                        for ci, cl in enumerate(LABELS):
                            r[f'p_{name}_{cl}'] = float(probs[anchor, ci])
                    records.append(r)
                    for idx in ids:
                        b = dict(fp_id=ident, block_id=f.loc[idx, 'block_id'], block_index=int(f.loc[idx, 'block_index']), start_s=float(f.loc[idx, 'start_s']), end_s=float(f.loc[idx, 'end_s']), truth=f.loc[idx, 'y_true'], is_anchor=bool(idx == anchor), dual_decision=LABELS[yd[idx]], geometry_decision=LABELS[yg[idx]], fused_decision=LABELS[y[idx]])
                        for name, probs in [('dual', dual), ('geometry', geom), ('fused', p)]:
                            for ci, cl in enumerate(LABELS):
                                b[f'p_{name}_{cl}'] = float(probs[idx, ci])
                        block_records.append(b)
    print(f'{fold}: all FP events audited', flush=True)
table = pd.DataFrame(records)
blocks = pd.DataFrame(block_records)
assert table.fp_id.is_unique and len(table) == sum((v['fp'] for v in checks))
assert blocks.groupby('fp_id').is_anchor.sum().eq(1).all()
table.to_csv(OUT / 'fp_eventos.csv', index=False, encoding='utf-8-sig')
blocks.to_csv(OUT / 'fp_bloques.csv', index=False, encoding='utf-8-sig')
summary = table.groupby(['fusion', 'policy', 'event_class', 'branch_support_at_anchor']).size().unstack(fill_value=0).reindex(columns=['Sólo dual', 'Sólo geometría', 'Ambas', 'Sólo combinación'], fill_value=0)
summary['Total'] = summary.sum(axis=1)
summary.to_csv(OUT / 'resumen_ramas.csv', encoding='utf-8-sig')
esc = lambda v: html.escape(str(v))

def stamp(s):
    return f'{int(s) // 60:02d}:{s % 60:04.1f}'
titles = {'argmax': 'Line argmax / scrum argmax', 'line_argmax_scrum_035': 'Line argmax / scrum 0,35', 'line_050_scrum_035': 'Line 0,50 / scrum 0,35', 'original_fold_thresholds': 'Umbrales originales por fold'}
sections = []
for (fusion, policy), g in table.groupby(['fusion', 'policy'], sort=False):
    heading = f"{('Geométrica' if fusion == 'geometric' else 'Aritmética')} · {titles[policy]}"
    items = []
    for r in g.itertuples():
        probrows = ''.join((f'<tr><th>{name}</th>' + ''.join((f"<td>{getattr(r, 'p_' + prefix + '_' + cl):.4f}</td>" for cl in LABELS)) + f'<td>{esc(getattr(r, decision))}</td></tr>' for name, prefix, decision in [('Dual', 'dual', 'dual_decision'), ('Geometría', 'geometry', 'geometry_decision'), ('Fusión', 'fused', 'fused_decision')]))
        br = blocks[blocks.fp_id.eq(r.fp_id)].copy()
        display = br[['block_index', 'start_s', 'end_s', 'truth', 'is_anchor', 'dual_decision', 'geometry_decision', 'fused_decision']].to_html(index=False, border=0)
        items.append(f"""<details class="event"><summary><b>{esc(r.fold)} · {esc(r.section)} · {stamp(r.start_s)}–{stamp(r.end_s)} · {r.event_class.upper()}</b><span>{esc(r.branch_support_at_anchor)}</span></summary><div class="inside"><p><b>Tipo de FP:</b> {esc(r.fp_type)}. <b>Etiqueta real en el ancla:</b> {esc(r.truth_at_anchor)}. <b>Alpha:</b> {r.alpha:.2f}.</p><p>Ancla: {stamp(r.anchor_s)}; bloque {r.anchor_block_index}. Distancia a la jugada real más cercana: {('sin eventos de esa clase en la sección' if pd.isna(r.nearest_truth_distance_s) else f'{r.nearest_truth_distance_s:.1f} s')}. Tiempos relativos a la sección.</p><table><thead><tr><th>Rama</th><th>P(line)</th><th>P(other)</th><th>P(scrum)</th><th>Decisión con la misma regla</th></tr></thead><tbody>{probrows}</tbody></table><p><b>¿El umbral cambia el argmax de la fusión en el ancla?</b> {('Sí' if r.threshold_changes_anchor_decision else 'No')}.</p><p><b>Usando sólo dual:</b> {esc(r.only_dual_event_status)} ({r.only_dual_overlapping_fp} FP, {r.only_dual_overlapping_tp} aciertos).<br><b>Usando sólo geometría:</b> {esc(r.only_geometry_event_status)} ({r.only_geometry_overlapping_fp} FP, {r.only_geometry_overlapping_tp} aciertos).</p><p>Bloques de esta detección apoyados por cada rama: dual {r.dual_support_blocks}/{r.event_blocks}; geometría {r.geometry_support_blocks}/{r.event_blocks}.</p>{display}<p class="muted">ID: {esc(r.fp_id)}</p></div></details>""")
    sections.append(f"""<details class="config"><summary>{esc(heading)} <small>({len(g)} FP)</small></summary>{''.join(items)}</details>""")
document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Origen de cada falso positivo · M1</title><style>body{font:16px system-ui,sans-serif;background:#f3f6fa;color:#1a2c43;margin:0}main{max-width:1250px;margin:auto;padding:35px}h1{font-size:32px}p{line-height:1.55}table{border-collapse:collapse;background:white;width:100%;font-size:14px;margin:14px 0}th,td{padding:8px;border:1px solid #dce3ed;text-align:left}th{background:#e8eff7}summary{cursor:pointer;padding:15px}.config{background:white;border:1px solid #cad5e4;margin:18px 0;border-radius:8px}.config>summary{font-size:19px;font-weight:600}.event{border-top:1px solid #e2e7ee;margin:0 15px}.event span{float:right;color:#34567c}.inside{padding:0 15px 15px}.muted{font-size:12px;color:#627187;overflow-wrap:anywhere}.note{background:#e5edf7;padding:18px;border-radius:8px}.scroll{overflow:auto}</style><main><h1>¿Qué rama sostuvo cada falso positivo?</h1><p>Seis modelos M1, cada uno en su test. Fusiones geométrica y aritmética, con el alpha original. No se recalcularon modelos.</p><div class="note"><b>Cómo leer la atribución:</b> se compara la decisión de cada rama por separado, con la misma regla, en el bloque de mayor probabilidad de la detección (ancla). <b>Sólo dual / sólo geometría</b>: únicamente esa rama sostiene la clase; <b>ambas</b>: las dos la sostienen; <b>sólo combinación</b>: ninguna por separado la sostiene.<br>Esto identifica apoyo a la decisión, no demuestra una causa única del error. Los FP duplicados pueden tener una clase correcta pero perder el emparejamiento uno a uno. La prueba con una sola rama vuelve a formar y emparejar todos los eventos; se muestran los que comparten bloques con el FP original. Una detección puede desaparecer, dividirse o unirse a otra.</div><h2>Resumen del apoyo en el ancla</h2><div class="scroll">' + summary.reset_index().replace({'geometric': 'Geométrica', 'arithmetic': 'Aritmética', **titles}).to_html(index=False, border=0) + '</div><h2>Detalle de cada FP</h2><p>Abrí una configuración y después cada detección para ver probabilidades, decisiones y pruebas con una sola rama. Los mismos intervalos pueden aparecer en varias configuraciones: no se deben sumar como errores únicos.</p>' + ''.join(sections) + '<p class="muted">Emparejamiento original: uno a uno, máxima recompensa temporal, tolerancia de 35 segundos. Fuentes: predictions.csv y run.json de PV-M1-H2-D2-PV01…PV06-S42/attempt_001. CSV detallados: fp_eventos.csv y fp_bloques.csv.</p></main></html>'
(OUT / 'informe_fp.html').write_text(document, encoding='utf-8')
assert all((sha(p) == h for p, h in hashes.items()))
assert not any((k == 'keras' or k.startswith('tensorflow') for k in sys.modules))
(HERE / 'verification.json').write_text(json.dumps(dict(passed=True, fp_rows=len(table), block_rows=len(blocks), configurations=8, folds=6, all_fp_counts_match_previous_comparison=True, source_sha256=hashes, one_anchor_per_fp=True, no_inference=True, attribution='Same-policy branch decisions at fused event anchor; full single-branch event matching also included'), indent=2), encoding='utf-8')
print(summary.to_string(), flush=True)
print('FP types:', table.groupby(['fusion', 'policy', 'fp_type']).size().to_string(), flush=True)
print(f'Created {len(table)} FP records and {len(blocks)} block records', flush=True)
