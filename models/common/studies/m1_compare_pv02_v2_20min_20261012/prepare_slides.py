"""Prepare a reviewed replacement for slide 44, after both timed runs pass."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from zipfile import ZipFile
import json, re, hashlib
B = experiment_directory(__file__)
ROOT = B.parents[1]
T = ROOT / 'docs/presentacion_guion_20260908/.build/v9_yolo_ablation_v2_20min'
T.mkdir(exist_ok=True)
for path in [B / 'results/M1-H2-D2_completion.json', B / 'results/completion.json']:
    d = json.loads(_resolve_model_artifact(path).read_text())
    assert d['all_checks_pass'] and d['hashes_unchanged']
full = json.loads(_resolve_model_artifact(B / 'results/M1-H2-D2_result.json').read_text())
pose = json.loads(_resolve_model_artifact(B / 'results/result.json').read_text())

def clock(t):
    v = round(t)
    return (f'{v // 3600} h ' if v >= 3600 else '') + f'{v % 3600 // 60} min {v % 60:02} s'
speech = f"Esta segunda comparación corresponde a otros veinte minutos: el primer tramo de V2. Usamos el entrenamiento PV02, que reservó V2 para prueba y V3 para desarrollo. Las dos variantes conservan los mismos pesos de esa partición.\n\nEl modelo completo tardó {clock(full['metadata']['timing']['inference_seconds'])}. La variante sólo YOLO tardó {clock(pose['metadata']['timing']['inference_seconds'])}. Cada corrida extrajo sus características desde el video.\n\nLas bandas permiten revisar coincidencias, omisiones y marcas fuera del etiquetado real. Los umbrales se eligieron con desarrollo. Estos tiempos corresponden a un único fold; la comparación anterior de VF utilizó seis. Los últimos tres segundos no tienen anotación."
source = ROOT / 'docs/presentacion_guion_20260908/output/presentacion_defensa_desde_guion_v9_en_desarrollo_r38.pptx'
g = ROOT / 'docs/guion_defensa_40_min_20261011.md'
old = _resolve_model_artifact(g).read_text(encoding='utf-8')
match = re.search('^## Diapositiva 44\\..*?(?=^## Diapositiva 45\\.)', old, re.S | re.M)
assert match
timing = re.search('Tiempo orientativo: (.*?) \\((.*?)\\)\\.', match.group()).groups()
cue = 'Comparar Real, RGB + YOLO y geometría sola en otros veinte minutos de V2. Aclarar PV02 único, V2 prueba y V3 desarrollo.'
replacement = f'## Diapositiva 44. V2 · PV02: predicciones con y sin RGB\n\nTiempo orientativo: {timing[0]} ({timing[1]}).\n\n[{cue}]\n\n{speech}\n\n'
updated = old[:match.start()] + replacement + old[match.end():]
updated = updated.replace('revisión 38', 'revisión 39')
updated = updated.replace('Las diapositivas 43 y 44 comparan Real, RGB + YOLO y geometría sola en VF_S01.', 'Las diapositivas 43 y 44 comparan Real, RGB + YOLO y geometría sola en VF_S01 y V2, respectivamente.')
updated = updated.replace('La segunda amplía 06:00–12:00. Los tiempos corresponden a los veinte minutos completos.', 'La segunda usa otros veinte minutos de V2 con el entrenamiento PV02. V2 quedó reservado para prueba. Sus tiempos corresponden a corridas nuevas de un fold, frente a los seis folds usados en VF_S01.')
parts = re.split('(?=^## Diapositiva \\d+\\.)', updated, flags=re.M)[1:]
words = 0
for part in parts:
    body = part.split('## Preparación para el ensayo', 1)[0]
    body = re.split('\\]\\n\\n', body, 1)[1]
    words += len(body.split())
updated = re.sub('El texto oral tiene [\\d.]+ palabras, cerca de [\\d.]+ minutos', f'El texto oral tiene {words:,}'.replace(',', '.') + f' palabras, cerca de {words / 125:.1f} minutos', updated)
(T / 'guion_anterior_r38.md').write_text(old, encoding='utf-8')
(T / 'guion_actualizado.md').write_text(updated, encoding='utf-8')
note = f'Diapositiva 44. V2 · PV02: predicciones con y sin RGB\nTiempo orientativo: {timing[0]}.\nINDICACIÓN VISUAL (no leer): {cue}\nTEXTO ORAL\n' + speech.replace('\n\n', '\n') + '\nAPOYO PARA PREGUNTAS (no leer):\nEntrenamiento PV-M1-H2-D2-PV02-S42, attempt_001. Entrenamiento V1/V4/V5/V6, desarrollo V3, prueba V2. Un modelo dual y un Random Forest; sin ajuste nuevo. No se vota con otros folds que entrenaron con V2.\nFuente exacta: ' + full['video_path'] + '\nReal: sesión manual 20260816_173319_f828c9, 171 bloques de 7 segundos, cobertura 0–1197 s. Duración del archivo 1199.8987 s; cola final sin etiqueta, no asumir clase otro. Los eventos reales agrupan bloques contiguos.\nPredicciones: inferencias nuevas, ventanas de 7 s y stride de 1 s, 25 FPS. ResNet + YOLO nuevos en completo; sólo YOLO nuevo en geometría. Cada rama usa su umbral de desarrollo de PV02: completo line/scrum 0.40/0.50, geometría 0.30/0.50. La ablación elimina ResNet y la rama recurrente dual, conservando el forest. No aísla sólo RGB.\nEl entrenamiento evaluó bloques de prueba de stride 7 s; esta nueva visualización agrupa eventos de ventanas solapadas de stride 1 s. No atribuir directamente el F1 publicado a este gráfico. Los tiempos incluyen extracción y clasificación; excluyen importación, carga y calentamiento. Son de un fold y no se comparan como aceleración directa con los tiempos de seis folds de la diapositiva 43.\nFuentes: D:/Tesina/data/m1_compare_pv02_v2_20min_20261012/inputs/campaign.json; results/*timing.json; figures/image_v2_pv02_evidence.json.'
payload = {'source': str(source), 'source_sha256': hashlib.sha256(_resolve_model_artifact(source).read_bytes()).hexdigest(), 'guion_source_sha256': hashlib.sha256(old.encode()).hexdigest(), 'image': str(B / 'figures/comparacion_real_inferencias_v2_pv02_20min.png'), 'note': note, 'times_seconds': {'full': full['metadata']['timing']['inference_seconds'], 'pose': pose['metadata']['timing']['inference_seconds']}}
(T / 'replacement.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print('Replacement ready: another annotated 20-minute source, actual new timings, one held-out fold.')
