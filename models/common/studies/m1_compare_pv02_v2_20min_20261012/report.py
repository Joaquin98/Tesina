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
full = json.loads(_resolve_model_artifact(B / 'results/M1-H2-D2_result.json').read_text())
pose = json.loads(_resolve_model_artifact(B / 'results/result.json').read_text())
source = json.loads(_resolve_model_artifact(B / 'inputs/campaign.json').read_text())

def clock(t):
    s = round(t)
    return (f'{s // 3600} h ' if s >= 3600 else '') + f'{s % 3600 // 60} min {s % 60:02} s'
tf = full['metadata']['timing']
tp = pose['metadata']['timing']
rows = []
for name, r, t in [('M1-H2-D2 completo · PV02', full, tf), ('Sólo YOLO + geometría · PV02', pose, tp)]:
    counts = {c: sum((e['play_type'] == c for e in r['detections'])) for c in ['line', 'scrum']}
    rows.append({'name': name, 'inference_seconds': t['inference_seconds'], 'events': counts, 'stage_seconds': t['stage_seconds'], 'counts': t['counts']})
summary = {'video': full['video_path'], 'seconds': full['metadata']['duration'], 'fold': 'PV02', 'rows': rows, 'saved_seconds': tf['inference_seconds'] - tp['inference_seconds'], 'observed_reduction_percent': 100 * (1 - tp['inference_seconds'] / tf['inference_seconds']), 'fresh_extraction': True, 'six_fold_VF_timings_are_not_directly_comparable': True}
(B / 'timing_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
text = f'# Otros veinte minutos: V2, entrenamiento PV02\n\nSe compararon M1-H2-D2 completo y su rama YOLO/geometría sobre el archivo exacto de los primeros veinte minutos de V2, con el único checkpoint PV02. V2 quedó reservado para prueba; V3 se usó para desarrollo; V1, V4, V5 y V6 para entrenamiento. Se conservaron arquitectura y pesos. La variante geométrica elimina ResNet y la cabeza recurrente dual, y conserva el Random Forest de PV02. No aísla exclusivamente RGB.\n\n| Variante | Inferencia completa del video | Lineouts predichos | Scrums predichos |\n|---|---:|---:|---:|\n'
for row in rows:
    text += f"| {row['name']} | {clock(row['inference_seconds'])} | {row['events']['line']} | {row['events']['scrum']} |\n"
text += f"\nEl ahorro observado fue {summary['observed_reduction_percent']:.2f} %. Ambas corridas fueron secuenciales, con extracción nueva desde video a 25 FPS y la misma CPU/hilos (TensorFlow 8/1, Torch 8/1, OpenCV 1). La rama recurrente completa usa un grafo con un trazado. La importación, carga y calentamiento se registraron por separado. Los tiempos de VF_S01 correspondían a seis folds; no se comparan directamente con estos de un único fold. La CPU no está reservada y los resultados son mediciones de estas corridas.\n\nLos umbrales line/scrum del modelo completo son 0.40/0.50 y los de geometría 0.30/0.50, seleccionados con V3. Se usan ventanas de 7 s con paso de 1 s y agrupación de eventos. Los F1 publicados de PV02 se calcularon sobre los bloques del protocolo original, por lo que no se atribuyen automáticamente a esta nueva grilla solapada. Los conteos de marcas no miden exactitud.\n\nReal proviene del etiquetado humano final de la sesión 20260816_173319_f828c9: 171 bloques consecutivos de siete segundos cubren 0–1197 s. Se verificó identidad de etiquetas e IDs contra los bloques de prueba PV02, además de CSV de revisión y candidatos de sesión. Se agruparon bloques consecutivos de la misma clase. Los últimos 2.9 s del archivo no tienen etiqueta; no se asume clase «other». El archivo dura 1199.8987 s, es decir, aproximadamente veinte minutos. Se utilizó el archivo exacto anotado, para evitar discrepancias entre codificaciones de S01.\n\n![Comparación completa](figures/comparacion_real_inferencias_v2_pv02_20min.png)\n\nArchivos: inputs/campaign.json; results/M1-H2-D2_result.json y result.json; timings y completion.json; eventos y ventanas CSV; figures/image_v2_pv02_evidence.json. Los hashes de entradas y pesos se verificaron antes y después de cada corrida. La inferencia parcial iniciada en VF_S02 se canceló al elegir este video anotado y no integra ningún tiempo completo.\n\nLa figura reemplaza la ampliación errónea de 06:00–12:00 en la slide 44 de la revisión 39. La comparación completa de VF_S01 sigue en la slide 43. El informe se limita a esta comparación.\n"
(B / 'README.md').write_text(text, encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
