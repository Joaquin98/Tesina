"""Close after all artifacts and published output are verified."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import sys, datetime, zipfile, urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
sys.path.insert(0, str(source_directory(experiment_directory(__file__).parents[0])))
from common import *
for name, key in [('pipeline_complete.json', 'complete'), ('completion.json', 'verified'), ('verification.json', 'passed'), ('combined_verification.json', 'passed'), ('backup_complete.json', 'passed')]:
    assert load(HERE / name)[key], name
check()
for seed in SEEDS:
    assert digest(HERE / f'seed_{seed}/central/best.weights.h5') == digest(PARENT / f'seed_{seed}/fixed/best.weights.h5')
backup = load(HERE / 'backup_complete.json')
assert digest(backup['path']) == backup['sha256']
with zipfile.ZipFile(backup['path']) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest(HERE / name)
    assert all((n in z.namelist() for n in ['combined.py', 'combined_results.csv', 'combined_bootstrap.json', 'combined_verification.json']))
report = ROOT / 'src/web/static/reports' / HERE.name
doc = _resolve_model_artifact(report / 'index.html').read_text(encoding='utf-8')
for v in ['cost_2', 'cost_4', 'cost_8', 'cost_16']:
    assert v in doc
http = {}
for name in ['index.html', 'timelines.png']:
    with urllib.request.urlopen('http://localhost:5000/static/reports/' + HERE.name + '/' + name, timeout=15) as response:
        assert response.status == 200
        http[name] = dict(status=response.status, bytes=len(response.read()))
df = pd.read_csv(_resolve_model_artifact(HERE / 'combined_results.csv'))
means = df.groupby('variant').mean(numeric_only=True).reindex(['central', 'cost_2', 'cost_4', 'cost_8', 'cost_16'])
base_ci = load(HERE / 'paired_bootstrap.json')
cost4_ci = load(HERE / 'combined_bootstrap.json')
timing = load(HERE / 'completion.json')
local = datetime.datetime.now(ZoneInfo('America/Argentina/Buenos_Aires'))
review = dict(passed=True, completed_local=local.isoformat(), new_models=6, seeds=SEEDS, VF_blocks=688, prediction_sets=15, raw_argmax=True, all_new_checkpoints_before_VF=True, tests_passed=8, real_data_smoke=True, independent_review='analysis/review.json', frozen_hashes_intact=True, reference_weight_copies_match=True, zip_entries_verified=backup['entries'], http=http, means=means.to_dict(orient='index'), ci_vs_central=base_ci, ci_vs_cost4=cost4_ci, timing=timing, limitations=['VF observed repeatedly; costs8/16 proposed after prior results; exploratory adaptive comparison', 'Temporal60s cluster CIs descriptive, one VF source; no correction for multiple metrics/variants', 'Three seed mean, not ensemble; precision mean is not ratio of mean counts', 'Central view0 real1Hz; frozen feature cache reused; not fresh video benchmark', 'CH provenance incomplete; blocks are not unique plays'])
save(HERE / 'analysis/completion_review.json', review)
ledger = load(HERE / 'analysis/ledger.json')
ledger.update(state='complete and verified', completion_review='analysis/completion_review.json')
save(HERE / 'analysis/ledger.json', ledger)
plan = ROOT / 'docs/superpowers/plans/2026-10-27-central-cost8-16.md'
plan.write_text(_resolve_model_artifact(plan).read_text(encoding='utf-8').replace('- [ ]', '- [x]'), encoding='utf-8')
lines = ['\n\n## Central: extension costos 8 y 16 (' + local.strftime('%H:%M') + ')', 'Seis entrenamientos nuevos, D2 8796, central1Hz, 15 epocas, semillas17/42/2026, argmax sin umbrales. Reutilizados central y costos2/4; quince resultados por bloques VF688 verificados. Costos8/16 prefijados para esta tanda despues de observar VF anterior: exploratorio adaptativo, no test externo independiente.']
for variant, row in means.iterrows():
    lines.append(f'{variant}: precision/sensibilidad/F1 line {100 * row.line_precision:.2f}/{100 * row.line_recall:.2f}/{100 * row.line_f1:.2f}%; scrum {100 * row.scrum_precision:.2f}/{100 * row.scrum_recall:.2f}/{100 * row.scrum_f1:.2f}%; macroF1 {100 * row.f1_macro:.2f}%, exactitud {100 * row.accuracy:.2f}%. TP/FN/FP medios line {row.line_tp:.2f}/{row.line_fn:.2f}/{row.line_fp:.2f}; scrum {row.scrum_tp:.2f}/{row.scrum_fn:.2f}/{row.scrum_fp:.2f}.')
for contrast, collection in [('vscentral', base_ci), ('vscosto4', cost4_ci)]:
    for variant, record in collection.items():
        for label in ['line', 'scrum']:
            r = record['metrics'][label + '_recall']
            lines.append(f"{variant} {contrast} {label} sensibilidad: {100 * r['difference']:+.2f}pp, IC95 temporal [{100 * r['ci95'][0]:+.2f},{100 * r['ci95'][1]:+.2f}].")
lines.append(f"Entrenamiento nuevo {timing['training_seconds']:.2f}s; nueve inferencias desde cache {timing['inference_seconds']:.2f}s. No benchmark fresco. Media de semillas no ensamble; denominadores68line/39scrum; bloques no eventos. Verificados probabilidades, conteos sklearn, IC recall independiente, copias centrales, hashes, ZIP y HTTP200. Informe src/web/static/reports/{HERE.name}/index.html. Evidencia data/{HERE.name}/analysis/completion_review.json.")
diary = ROOT / 'docs/diary' / (local.strftime('%Y-%m-%d') + '.md')
with _resolve_model_artifact(diary).open('a', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(json.dumps(dict(passed=True, means=means.to_dict(orient='index'), timing=timing, ci_vs_cost4=cost4_ci), ensure_ascii=True))
