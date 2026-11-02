"""Explain each argmax FP of the all-development M1 on VF, after replay check."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, html
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
verification = json.loads(_resolve_model_artifact(HERE / 'replay_verification.json').read_text())
assert verification['passed']
raw = json.loads(_resolve_model_artifact(HERE / 'branch_probabilities.json').read_text())
lookup = {r['block_id']: r for r in raw}
fp = pd.read_csv(_resolve_model_artifact(HERE / 'fp_inventory.csv'))
fp = fp[fp.policy.eq('argmax')].copy()
assert len(fp) == 11
labels = ['line', 'other', 'scrum']
a = verification['alpha']
rows = []

def stamp(s):
    return f'{int(s) // 60:02d}:{s % 60:04.1f}'
for i, e in enumerate(fp.sort_values(['section', 'anchor_s']).itertuples(), 1):
    b = lookup[e.anchor_block_id]
    d = np.clip(b['dual'], 1e-07, 1)
    g = np.clip(b['geometry'], 1e-07, 1)
    p = np.array(b['fused_original'])
    k = labels.index(e.label)
    assert p.argmax() == k
    competitor = max((j for j in range(3) if j != k), key=lambda j: p[j])
    dc = float((1 - a) * np.log(d[k] / d[competitor]))
    gc = float(a * np.log(g[k] / g[competitor]))
    assert abs(dc + gc - np.log(p[k] / p[competitor])) < 0.0002
    r = dict(fp_number=i, section=e.section, start_s=e.start_s, end_s=e.end_s, anchor_s=e.anchor_s, label=e.label, truth_at_anchor=e.truth_at_anchor, fp_type=e.fp_type, nearest_truth_distance_s=e.nearest_truth_distance_s, competitor=labels[competitor], dominant_branch='Dual' if dc > gc else 'Geometría', dual_contribution=dc, geometry_contribution=gc, combined_margin=dc + gc, dual_decision=labels[np.argmax(d)], geometry_decision=labels[np.argmax(g)], dual_supports_fp=bool(np.argmax(d) == k), geometry_supports_fp=bool(np.argmax(g) == k), alpha=a, replay_max_error=b['max_absolute_error'], block_id=e.anchor_block_id)
    for name, probs in [('dual', b['dual']), ('geometry', b['geometry']), ('fused', b['fused_original'])]:
        for j, c in enumerate(labels):
            r['p_' + name + '_' + c] = probs[j]
    rows.append(r)
table = pd.DataFrame(rows)
table.to_csv(OUT / 'atribucion_cada_fp.csv', index=False, encoding='utf-8-sig')
summary = table.groupby(['label', 'dominant_branch']).size().unstack(fill_value=0)
summary.to_csv(OUT / 'resumen.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(13, 8), facecolor='#f4f7fb')
ax.set_facecolor('#f4f7fb')
y = np.arange(len(table))
ax.barh(y - 0.17, table.dual_contribution, height=0.32, color='#2466ab', label='Red dual · peso 0,525')
ax.barh(y + 0.17, table.geometry_contribution, height=0.32, color='#cf7a2c', label='Geometría RF · peso 0,475')
ax.set_yticks(y, [f'FP {r.fp_number:02d} · {r.section} {stamp(r.anchor_s)} · {r.label}' for r in table.itertuples()])
ax.invert_yaxis()
ax.axvline(0, color='#38475d', lw=1)
ax.set_xlabel('Aporte al margen frente a la segunda clase de la fusión (log-odds)\nPositivo: favorece la clase detectada · Negativo: se opone')
ax.set_title('M1 entrenado con todos los datos · VF · ambos argmax\nInfluencia de cada rama en los 11 falsos positivos', loc='left', fontsize=17, fontweight='bold', pad=20)
ax.legend(loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.15)
ax.set_axisbelow(True)
fig.text(0.04, 0.02, 'Análisis en el bloque de máxima confianza de cada FP. S01–S04; tiempos relativos a cada sección.\nSe reprodujeron las salidas originales con los mismos pesos. Los duplicados también se cuentan como FP.', fontsize=10, color='#53647a')
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(OUT / 'influencia_ramas.png', dpi=145)
plt.close(fig)
esc = lambda v: html.escape(str(v))
cards = []
for r in table.itertuples():
    probrows = ''.join(('<tr><th>' + name + '</th>' + ''.join((f"<td>{getattr(r, 'p_' + prefix + '_' + c):.4f}</td>" for c in labels)) + '<td>' + esc(decision) + '</td></tr>' for name, prefix, decision in [('Dual', 'dual', r.dual_decision), ('Geometría', 'geometry', r.geometry_decision), ('Fusión', 'fused', r.label)]))
    cards.append(f'<details><summary><b>FP {r.fp_number:02d} · {r.section} · {stamp(r.start_s)}–{stamp(r.end_s)} · {r.label.upper()}</b><span>Mayor aporte: {r.dominant_branch}</span></summary><div class="body"><p><b>Tipo:</b> {esc(r.fp_type)}. <b>Etiqueta real del bloque:</b> {r.truth_at_anchor}. <b>Ancla:</b> {stamp(r.anchor_s)}. Distancia al evento real más cercano: {r.nearest_truth_distance_s:.1f} s.</p><table><tr><th>Modelo</th><th>P(line)</th><th>P(other)</th><th>P(scrum)</th><th>Argmax</th></tr>{probrows}</table><p>Para decidir <b>{r.label}</b> frente a <b>{r.competitor}</b>, la dual aporta <b>{r.dual_contribution:+.4f}</b> y geometría <b>{r.geometry_contribution:+.4f}</b>. Suma: {r.combined_margin:+.4f}. El aporte positivo mayor corresponde a <b>{r.dominant_branch}</b>.</p><p><b>Usando sólo dual, este bloque sería:</b> {r.dual_decision}. <b>Usando sólo geometría:</b> {r.geometry_decision}. Esta comprobación corresponde al ancla; no se recalcularon eventos completos de las ramas aisladas.</p><p class="small">Error máximo de reproducción: {r.replay_max_error:.2g} · Bloque: {esc(r.block_id)}</p></div></details>')
display = pd.DataFrame([{'FP': r.fp_number, 'Sección': r.section, 'Ancla': stamp(r.anchor_s), 'Clase': r.label, 'Mayor aporte': r.dominant_branch, 'Dual': round(r.dual_contribution, 4), 'Geometría': round(r.geometry_contribution, 4)} for r in table.itertuples()])
document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FP del M1 único en VF · Ambos argmax</title><style>body{margin:0;background:#f4f7fb;color:#192d47;font:16px system-ui}main{max-width:1120px;margin:auto;padding:30px}h1{font-size:30px}p{line-height:1.5}.note{background:#e4edf8;padding:18px;border-radius:8px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white;margin:14px 0}td,th{padding:9px;text-align:left;border:1px solid #d3deeb}th{background:#e7eef7}details{background:white;border:1px solid #cbd7e5;border-radius:7px;margin:12px 0}summary{cursor:pointer;padding:16px}summary span{float:right;color:#225f9d}.body{padding:0 18px 16px}.small{font-size:12px;color:#586d83}</style><main><h1>Qué modelo influyó más en cada FP del video final</h1><p><b>M1 único entrenado con CH + V1–V6 · VF S01–S04 · Ambos argmax.</b><br>Fusión geométrica: dual 0,525 + geometría 0,475. Son 9 falsos positivos de line y 2 de scrum.</p><div class="note">El aporte se mide en el bloque de mayor confianza de cada detección, frente a la segunda clase de la fusión:<br><b>Dual:</b> 0,525 × log[Pdual(clase detectada) / Pdual(competidora)]<br><b>Geometría:</b> 0,475 × log[Pgeom(clase detectada) / Pgeom(competidora)]<br>Los aportes suman el margen logarítmico de la fusión. Un aporte positivo favorece la decisión; uno negativo se opone. La rama dual ya combina RGB y pose; geometría es el Random Forest. Esta atribución explica la decisión en el ancla, no asigna una causa única a todo el evento. En un FP duplicado la clase puede ser correcta y el error estar en el agrupamiento o el emparejamiento.</div><img src="influencia_ramas.png" alt="Aporte de cada rama a los once FP"><h2>Los 11 FP</h2>' + display.to_html(index=False, border=0) + '<h2>Probabilidades y detalle por detección</h2>' + ''.join(cards) + f"""<p class="small">Las salidas se calcularon mediante inferencia de los 11 bloques ancla con los pesos originales, sin reentrenar. Se verificó cada fusión contra las probabilidades guardadas; error máximo {verification['max_error']:.3g}. Se conservaron las detecciones y el emparejamiento original, con tolerancia de 35 s. Tiempos relativos a cada sección; S05 excluida.</p></main></html>"""
(OUT / 'informe_vf_argmax.html').write_text(document, encoding='utf-8')
print(display.to_string(index=False))
print(summary.to_string())
print('Max replay error:', verification['max_error'])
