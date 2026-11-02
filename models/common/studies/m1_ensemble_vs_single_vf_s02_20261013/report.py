"""Report only measured VF_S02 values, with reuse and protocol made explicit."""
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
B = experiment_directory(__file__)

def clock(t):
    v = round(t)
    return (f'{v // 3600} h ' if v >= 3600 else '') + f'{v % 3600 // 60} min {v % 60:02} s'
req = json.loads(_resolve_model_artifact(B / 'inputs/request.json').read_text(encoding='utf-8'))
rows = []
for key, name in [('M1-ensemble', 'Ensemble M1-H2-D2, seis folds'), ('M1-single', 'Modelo único M1-H2-D2, 8796 ejemplos')]:
    t = json.loads(_resolve_model_artifact(B / f'results/{key}_timing.json').read_text())
    q = json.loads(_resolve_model_artifact(B / f'results/{key}_quality.json').read_text())
    m = q['temporal']
    rows.append({'key': key, 'name': name, 'inference_seconds': t['inference_seconds'], 'imports_seconds': t['imports_seconds'], 'load_seconds': t['model_load_seconds'], 'warmup': t['warmup'], 'stage_seconds': t['stage_seconds'], 'counts': t['counts'], 'f1_line': m['line']['weighted_f1'], 'f1_scrum': m['scrum']['weighted_f1'], 'f1_mean': m['weighted_f1_macro'], 'far_FP_per_hour': sum((m[c]['far_false_positives_per_hour'] for c in ['line', 'scrum'])), 'quality': m, 'quality_seconds': q['quality_seconds']})
human = json.loads(_resolve_model_artifact(Path(req['human_session'])).read_text(encoding='utf-8'))
semantic = [{k: c[k] for k in ['candidate_index', 'start_time', 'end_time', 'final_label']} for c in human['candidates']]
unchanged = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest() == req['human_label_sha256']
assert unchanged, 'Human labels changed after capture: evaluate the revised labels before delivering.'
summary = {'video': 'VF_S02', 'video_seconds': 1201.2, 'models_reused_without_retraining': True, 'rows': rows, 'single_observed_reduction_percent': 100 * (1 - rows[1]['inference_seconds'] / rows[0]['inference_seconds']), 'human_labels_unchanged': unchanged, 'independent_F1_claim_allowed': False, 'parameters_selected_using_VF': False}
with _resolve_model_artifact(B / 'results/comparison.json').open('x', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
text = '# VF_S02: ensemble y modelo único M1-H2-D2\n\nSe reutilizaron los mismos seis miembros del ensemble y el mismo modelo único entrenado con 8796 ejemplos de desarrollo de la comparación anterior en VF_S01. No se entrenó otra red ni otro forest. Los umbrales lineout/scrum (0,50/0,45) y la fusión quedaron fijados con desarrollo; no se ajustaron con este segmento.\n\nReal proviene de la sesión humana 20261013_102705_9a03b2: 172 bloques que cubren 1201,2 segundos, incluido el último de 4,2 s. Se congeló una copia de las etiquetas completas al preparar la prueba y se verificó que sus clases no cambiaran. No se exigió exportar clips ni se modificó el dataset para poder evaluar.\n\n| Alternativa | Inferencia desde video | F1 lineout | F1 scrum | F1 medio | FP lejanos/h |\n|---|---:|---:|---:|---:|---:|\n'
for r in rows:
    text += f"| {r['name']} | {clock(r['inference_seconds'])} | {r['f1_line']:.4f} | {r['f1_scrum']:.4f} | {r['f1_mean']:.4f} | {r['far_FP_per_hour']:.2f} |\n"
text += f"\nReducción observada del tiempo del modelo único: {summary['single_observed_reduction_percent']:.2f} %. Cada recorrido secuencial vuelve a decodificar el video y a aplicar ResNet-50 y YOLO a 30030 cuadros a 25 FPS, sin reutilizar características del otro modelo ni un caché persistente. Las 1195 ventanas de inferencia duran 7 s y avanzan cada 1 s. Ambos usan grafo y los mismos hilos de CPU (TF 8/1, Torch 8/1, OpenCV 1).\n\n"
text += '| Alternativa | Importación | Carga | Calentamiento RGB/pose/cabezas | Evaluación posterior |\n|---|---:|---:|---|---:|\n'
for r in rows:
    text += f"| {r['name']} | {r['imports_seconds']:.2f} s | {r['load_seconds']:.2f} s | " + ' / '.join((f'{v:.2f} s' for v in r['warmup'].values())) + f" | {r['quality_seconds']:.2f} s |\n"
text += '\nImportación, carga, calentamiento y evaluación posterior quedan fuera del cronómetro de inferencia. Para el F1 de la tesina se clasifican los 172 bloques humanos usando las características nuevas capturadas en RAM de esa misma corrida, después del cronómetro. Se agrupan bloques contiguos, con centro del evento real y centro del primer bloque de mayor probabilidad predicha; se aplica emparejamiento global uno a uno con tolerancia de 35 s y crédito max(0,1−distancia/35). La imagen corresponde a esas decisiones y al F1 registrado.\n\nLa CPU no está reservada; la variación de extracción y el costo de advertencias de sklearn, si aparecen, forman parte del tiempo observado. No se atribuye toda diferencia temporal al número de miembros. Las etapas instrumentadas pueden anidarse y no se suman como componentes independientes.\n\nEstos F1 son descriptivos: la independencia de todos los clips históricos respecto del partido final no está completamente verificada. No se usaron estas etiquetas para entrenar ni elegir hiperparámetros. Se preservaron modelos, fuentes, mediciones anteriores, presentación y tesina; los hashes de cada corrida están en sus archivos completion.\n\n![Real, ensemble y modelo único durante veinte minutos](figures/comparacion_ensemble_modelo_unico_20min_v2.png)\n\nFuentes reproducibles: inputs/request.json, human_session.json, human_blocks.csv, manifiestos y política de desarrollo; results/*_timing.json, *_quality.json, *_quality_blocks.csv, *_quality_events.csv, *_completion.json; figures/image_evidence_v2.json y final_validation.json.\n'
with _resolve_model_artifact(B / 'README.md').open('x', encoding='utf-8') as f:
    f.write(text)
print(json.dumps({'video': 'VF_S02', 'models_reused': True, 'rows': [{'name': r['name'], 'seconds': r['inference_seconds'], 'F1': r['f1_mean']} for r in rows]}))
