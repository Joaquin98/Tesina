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
import sys, json
import cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
OUT = experiment_directory(__file__)
sys.path.insert(0, str(source_directory(ROOT / 'data/multicrop_d2_stride25_vf_20261026')))
from common import views
source = ROOT / 'data/videos/partidos_evaluacion/VF_S01_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4'
cap = cv2.VideoCapture(str(source))
assert cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS)
idx = round(1088.5 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
ok, frame = cap.read()
cap.release()
assert ok and frame.shape[:2] == (720, 1280)
crops, meta = views(frame)
assert len(meta) == 9 and all((tuple(c.shape) == (224, 224, 3) for c in crops))
assert all((m['resize'] == [640, 360] for m in meta))
img = cv2.cvtColor(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
palette = {0: '#ffffff', 1: '#f77f00', 2: '#2ef2d0', 3: '#ffcc26', 4: '#dc5ce6', 5: '#ff5f70', 6: '#66b2ff', 7: '#ff8c31', 8: '#bdff54'}
selected = [2, 3, 6, 7, 0]
fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='#f5f7f7')
fig.suptitle('Un mismo fotograma: dónde mira cada recorte', fontsize=20, fontweight='bold', y=0.975)
for ax, k in zip(axes.flat, [0, 2, 3, 6, 7, None]):
    ax.imshow(img)
    ax.axis('off')
    if k is None:
        ax.set_visible(False)
        continue
    x, y, w, h = meta[k]['rect']
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=palette[k], linewidth=3.5))
    ax.set_title('Central 0' if k == 0 else f'Contexto {k}', fontweight='bold', fontsize=13)
axes.flat[5].set_visible(True)
axes.flat[5].imshow(img)
axes.flat[5].axis('off')
axes.flat[5].set_title('Los cinco superpuestos', fontweight='bold', fontsize=13)
for k in selected:
    x, y, w, h = meta[k]['rect']
    axes.flat[5].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=palette[k], linewidth=2.5, linestyle='--' if k == 0 else '-'))
fig.text(0.5, 0.028, 'VF S01 · 18:08,5 · línea blanca punteada: central · contornos de color: contextos 2, 3, 6 y 7', ha='center', fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT / 'vf_superposicion_5_recortes.png', dpi=160, facecolor=fig.get_facecolor())
plt.close(fig)
fig, axes = plt.subplots(3, 3, figsize=(11, 11), facecolor='#f5f7f7')
fig.suptitle('Las nueve imágenes del mismo fotograma', fontsize=19, fontweight='bold', y=0.98)
for k, ax in enumerate(axes.flat):
    ax.imshow(cv2.cvtColor(crops[k], cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('0 · CENTRAL' if k == 0 else f'{k} · contexto', fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(3 if k in [0, 2, 3, 6, 7] else 1)
        spine.set_color(palette[k] if k in [0, 2, 3, 6, 7] else '#9aa5a7')
fig.text(0.5, 0.025, 'Contornos destacados: la propuesta central + 2, 3, 6 y 7. Cada vista mide 224 × 224 píxeles.', ha='center', fontsize=10)
fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
fig.savefig(OUT / 'vf_nueve_recortes_reales.png', dpi=160, facecolor=fig.get_facecolor())
plt.close(fig)
(OUT / 'vf_ejemplo_recortes.json').write_text(json.dumps({'source': str(source), 'source_frame': idx, 'source_fps': fps, 'actual_seconds': idx / fps, 'frame_size': [1280, 720], 'normalized_size': [640, 360], 'geometry': meta, 'selected': [0, 2, 3, 6, 7]}, indent=2), encoding='utf8')
print('frame', idx, 'fps', fps, 'rectangles', [(x['view'], x['rect']) for x in meta])
