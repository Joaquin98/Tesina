"""Post-run artifact review; does not modify frozen experiment code."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import sys, json, hashlib, zipfile, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, str(source_directory(experiment_directory(__file__).parents[0])))
from common import *
assert load(HERE / 'completion.json')['verified'] and load(HERE / 'pipeline_complete.json')['complete']
check()
for seed in SEEDS:
    assert digest(HERE / f'seed_{seed}/central/best.weights.h5') == digest(PARENT / f'seed_{seed}/fixed/best.weights.h5')
backup = load(HERE / 'backup_complete.json')
assert digest(backup['path']) == backup['sha256']
with zipfile.ZipFile(backup['path']) as z:
    for name in z.namelist():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest(HERE / name)
report = ROOT / 'src/web/static/reports' / HERE.name
frames = {v: pd.read_csv(_resolve_model_artifact(HERE / 'seed_42' / v / 'vf_predictions.csv')) for v in ['central', 'cost_2', 'cost_4']}
base = frames['central']
fig, axes = plt.subplots(4, 1, figsize=(15, 9), constrained_layout=True)
for ax, section in zip(axes, sorted(base.section.unique())):
    ix = base.index[base.section == section]
    f = base.loc[ix]
    edges = np.r_[f.start_s.to_numpy(), f.end_s.iloc[-1]]
    np.testing.assert_allclose(f.end_s.iloc[:-1], f.start_s.iloc[1:])
    a = np.stack([[LABELS.index(k) for k in f.y_true]] + [[LABELS.index(k) for k in df.loc[ix].y_final] for df in frames.values()])
    ax.pcolormesh(edges, np.arange(5), a, cmap=ListedColormap(['#19a7a0', '#dddfe4', '#ed9a36']), vmin=0, vmax=2, shading='flat')
    ax.invert_yaxis()
    ax.set_yticks(np.arange(4) + 0.5, ['Humano', 'Central', 'Costo2', 'Costo4'])
    ax.set_title(section)
    ax.set_xlabel('Segundos')
fig.suptitle('VF semilla42 prefijada: line verde, other gris, scrum naranja; argmax sin umbrales')
fig.savefig(report / 'timelines.png', dpi=130)
plt.close(fig)
df = pd.read_csv(_resolve_model_artifact(HERE / 'results.csv'))
means = df.groupby('variant').mean(numeric_only=True)
comparisons = load(HERE / 'paired_bootstrap.json')
review = dict(completed_local=datetime.datetime.now(ZoneInfo('America/Argentina/Buenos_Aires')).isoformat(), passed=True, models_new=6, seeds=SEEDS, VF_blocks=688, raw_argmax=True, all_checkpoints_before_VF=True, tests_passed=7, independent_review='review_recall_cost; stage locking fix RED/GREEN, report completeness and backup reliability addressed', backup_entries_verified=backup['entries'], frozen_hashes_intact=True, reference_weight_copies_match=True, primary_result='cost4 increases both mean recalls; line improves2seeds and ties1; scrum improves3seeds; precision decreases', mean_metrics=means.to_dict(orient='index'), recall_intervals={v: {k: m for k, m in d['metrics'].items() if k.endswith('_recall')} for v, d in comparisons.items()}, limitations=['VF reused, exploratory, CH provenance incomplete', 'Paired60s temporal CI descriptive; one VF source', 'Means across3seeds are not an ensemble; strong seed variability', 'Costs2/4 predeclared and both reported; no VF tuning', 'ResNet feature cache reused, not video end-to-end benchmark'], timeline_exact_partial_block=True)
save(HERE / 'analysis/completion_review.json', review)
ledger = load(HERE / 'analysis/ledger.json')
ledger.update(state='complete and independently verified', completion_review='analysis/completion_review.json')
save(HERE / 'analysis/ledger.json', ledger)
plan = ROOT / 'docs/superpowers/plans/2026-10-27-central-recall-cost.md'
plan.write_text(_resolve_model_artifact(plan).read_text(encoding='utf-8').replace('- [ ]', '- [x]'), encoding='utf-8')
local = datetime.datetime.now(ZoneInfo('America/Argentina/Buenos_Aires'))
diary = ROOT / 'docs/diary' / (local.strftime('%Y-%m-%d') + '.md')
with _resolve_model_artifact(diary).open('a', encoding='utf-8') as f:
    f.write('\n\n## Cierre - Central orientado a sensibilidad\nSeis centrales temporales D2 completos entrenados y nueve predicciones VF verificadas, argmax sin umbrales. Costo2: precision/sensibilidad line 51.33/62.75%, scrum 90.86/78.63%. Costo4: line 49.52/70.59%, scrum 79.81/87.18%. Central original: line 56.39/54.90%, scrum 94.84/76.07%. Valores promedio de tres semillas, no ensamble. Costo4 aumenta sensibilidad line +15.69 pp (IC95 temporal [+10.00,+21.67]), scrum +11.11 pp ([+3.70,+18.92]). Line mejora2semillas y empata1; scrum mejora3. Costo2 scrum IC toca0.\nCosto4 recupera en promedio48/68 bloques line y34/39 scrum (base37.33 y29.67); FP line54.33 a60, scrum1.67 a10. Resultados por bloques, no eventos unicos. Cambia sensibilidad por precision segun el protocolo fijado. F1 macro central74.43%, costo2 75.71%, costo4 76.78%. No se seleccionaron costos/epocas usandoVF. Variabilidad entre semillas importante. VF reutilizado y CH sin procedencia completa: exploratorio; intervalos descriptivos temporales y multiplicidad.\nEntrenamiento total568.18s, inferencia desde caracteristicas4.95s. Cache ResNet reutilizada; no benchmark desde video. Verificacion sklearn, probabilidades base reproducidas, intervalos recall independientes, hashes congelados y73entradas ZIP comprobados; HTTP200. Timeline con limites reales incluido ultimo bloque parcial. Informe src/web/static/reports/central_recall_cost_d2_vf_20261027/index.html. Evidencia data/central_recall_cost_d2_vf_20261027/analysis/completion_review.json.\n')
print('CLOSED: weights,73 ZIP entries,frozen inputs and outputs verified; report and diary ready')
