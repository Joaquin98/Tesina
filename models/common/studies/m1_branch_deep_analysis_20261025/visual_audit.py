import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models.common.spatial import training_normalization_geometry
HERE = experiment_directory(__file__)
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
rows = pd.read_csv(_resolve_model_artifact(HERE / 'vf/anchor_events.csv'))
for status in ['FP', 'FN', 'TP']:
    subset = rows[rows.status.eq(status)].sort_values(['section', 'anchor_s'])
    if status == 'TP':
        subset = subset.groupby(['section', 'label'], sort=False).head(1)
    fig, axes = plt.subplots(int(np.ceil(len(subset) / 3)), 3, figsize=(16, 3.3 * np.ceil(len(subset) / 3)), squeeze=False, facecolor='#f4f7fb')
    for ax in axes.ravel():
        ax.axis('off')
    for ax, r in zip(axes.ravel(), subset.itertuples()):
        video = list(_model_artifact_glob(ROOT / 'data/videos/partidos_evaluacion', f'VF_{r.section}_*.mp4', recursive=False))
        assert len(video) == 1
        cap = cv2.VideoCapture(str(video[0]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(r.anchor_s * fps))
        ok, frame = cap.read()
        cap.release()
        assert ok
        h, w = frame.shape[:2]
        geo = training_normalization_geometry(w, h)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.add_patch(Rectangle((geo['source_roi_x'], geo['source_roi_y']), geo['source_roi_width'], geo['source_roi_height'], fill=False, edgecolor='#ffe545', lw=2))
        ax.set_title(f'{r.section} {int(r.anchor_s) // 60:02d}:{r.anchor_s % 60:04.1f} · {r.label} · real: {r.truth_at_anchor}', fontsize=10)
    fig.suptitle(f'VF · {status} · fotograma del bloque diagnóstico\nRectángulo amarillo: campo de visión que recibe M1', fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / f'vf_{status.lower()}_frames.png', dpi=145)
    plt.close(fig)
print('Visual audit sheets saved')
