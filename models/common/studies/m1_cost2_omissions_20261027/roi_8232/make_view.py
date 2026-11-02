"""Evidence-only spatial inspection using the recorded inference geometry."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, shutil, urllib.parse
HERE = experiment_directory(__file__)
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models.common.spatial import normalize_training_frame, training_normalization_geometry
frame = cv2.imread(str(HERE / 'source_8232.png'))
crop = cv2.imread(str(HERE / 'rgb_224_8232.png'))
g = training_normalization_geometry(frame.shape[1], frame.shape[0])
assert g == json.loads(_resolve_model_artifact(HERE / 'geometry.json').read_text())['geometry']
assert (normalize_training_frame(frame) == crop).all()
fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={'width_ratios': [16 / 9, 1]}, layout='constrained')
axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axes[0].add_patch(Rectangle((416, 136), 448, 448, fill=False, edgecolor='#ffdf00', linewidth=3))
axes[0].set_title('82:32 · Fotograma completo y región usada por M1', fontsize=15, pad=12)
axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), interpolation='nearest')
axes[1].set_title('Recorte RGB real · 224 × 224', fontsize=15, pad=12)
for ax in axes:
    ax.axis('off')
fig.savefig(HERE / 'comparison_8232.png', dpi=150, facecolor='white')
plt.close(fig)
WEB = ROOT / 'src/web/static/reports/m1_cost2_omissions_20261027/roi_8232'
WEB.mkdir(parents=True, exist_ok=True)
for name in ('comparison_8232.png', 'rgb_224_8232.png', 'pose_640_8232.png', 'source_8232.png', 'geometry.json'):
    shutil.copy2(HERE / name, WEB / name)
result = json.loads(_resolve_model_artifact(ROOT / 'data/results/result_20261027_argentina_allblacks_20250816_m1_cost2_stride1.json').read_text(encoding='utf-8'))
url = '/api/video-by-path?video_path=' + urllib.parse.quote(result['video_path'], safe='')
doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 · encuadre del line en 82:32</title><style>body{font:17px system-ui;line-height:1.5;max-width:1280px;margin:24px auto;padding:0 16px;color:#243447}img{max-width:100%;height:auto}.stage{position:relative;background:#000;line-height:0}video{display:block;width:100%;height:auto}.model-roi-overlay{display:none;position:absolute;box-sizing:border-box;border:3px solid #ffdf00;pointer-events:none}.model-roi-overlay.is-visible{display:block}.model-roi-overlay span{position:absolute;top:0;left:0;background:#111c;color:#ffdf00;font:12px system-ui;line-height:1.4;padding:4px;max-width:90%}button{padding:9px 14px;margin:6px 8px 6px 0;cursor:pointer}a{color:#126b84}aside{background:#edf3f8;padding:14px}</style><h1>Qué ve M1 en 82:32</h1><p>Argentina–Nueva Zelanda, 16/08/2025. El tiempo corresponde al video; el reloj del partido en este fotograma marca 64:45.</p><aside>La formación aparece en la zona inferior del recorte central fijo. Parte del sector próximo al touch queda fuera. El recuadro no sigue a los jugadores. Esto puede contribuir a la omisión, pero este fotograma por sí solo no demuestra su causa: M1 clasifica secuencias de 7 segundos.</aside><h2>Fotograma completo frente al recorte</h2><img src="comparison_8232.png" alt="Fotograma de 82:32 con borde amarillo del recorte y entrada RGB real de 224 por 224 al lado"><p>Fuente: 1280×720. Región común a RGB y pose: x=416…864, y=136…584, tamaño 448×448. RGB recibe 224×224; YOLO recibe esa misma región a 640×640. Imágenes reconstruidas con el preprocesamiento original, sin ejecutar los extractores ni cambiar predicciones.</p><h2>Revisar el video con el recuadro</h2><p><button id="still">Pausar en 82:32</button><button id="segment">Reproducir el segmento 82:22–82:43</button><label><input id="toggle" type="checkbox" checked> Mostrar recuadro del modelo</label></p><div id="stage" class="stage"><video id="video" controls preload="metadata"></video><div id="roi" class="model-roi-overlay" aria-hidden="true"><span></span></div></div><p id="status">Cargando 82:32…</p><p><a href="/compare?comparison=20261027_argentina_allblacks_20250816_human_m1x2_omissions#omisiones">Volver a las omisiones de M1 ×2</a> · <a href="rgb_224_8232.png">RGB 224×224</a> · <a href="pose_640_8232.png">Entrada visual de YOLO 640×640</a></p><script src="/static/js/model-roi-overlay.js"></script><script>\nconst video=document.getElementById(\'video\'),status=document.getElementById(\'status\');let stopAt=null;\nconst roiController=ModelRoiOverlay.attach({video,stage:document.getElementById(\'stage\'),overlay:document.getElementById(\'roi\'),toggle:document.getElementById(\'toggle\'),inputMode:\'source_equivalent_roi\'});\nconst tm=s=>Math.floor(s/60)+\':\'+String(Math.floor(s%60)).padStart(2,\'0\');\nfunction still(){stopAt=null;video.pause();video.currentTime=4952;}\nvideo.addEventListener(\'loadedmetadata\',still,{once:true});video.addEventListener(\'timeupdate\',()=>{status.textContent=\'Tiempo del video: \'+tm(video.currentTime);if(stopAt!==null&&video.currentTime>=stopAt){video.pause();stopAt=null;}});\ndocument.getElementById(\'still\').onclick=still;document.getElementById(\'segment\').onclick=()=>{video.currentTime=4939;stopAt=4966;video.play().catch(()=>{status.textContent=\'Presioná reproducir en el video.\';});};\nvideo.src=' + json.dumps(url) + ';\n</script></html>'
(WEB / 'index.html').write_text(doc, encoding='utf-8')
print('Created spatial evidence view without feature inference')
