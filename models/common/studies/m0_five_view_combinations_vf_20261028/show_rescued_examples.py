import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json
import numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
OUT = experiment_directory(__file__)
sys.path.insert(0, str(source_directory(ROOT / 'data/multicrop_d2_stride25_vf_20261026')))
from common import views
f = pd.read_csv(_resolve_model_artifact(ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027/vf_predictions.csv'))
labels = ['line', 'other', 'scrum']
colors = {0: '#e9ecef', 2: '#2ef2d0', 3: '#ffd23f', 6: '#66b2ff', 7: '#ff8c31'}
evidence = []
for case, (section, start, offset) in enumerate([('S04', 1085, 5), ('S01', 847, 1)], 1):
    row = f[(f.section == section) & (f.start_s == start)]
    assert len(row) == 1
    ix = row.index[0]
    item = row.iloc[0]
    probs = _load_model_result_array(ROOT / f'data/m0_h2_d2_allframes_nine_vf_20261027/blocks/{ix:04d}.npy')
    model = probs[1]
    central = model[0].argmax()
    vote = [i for i in [2, 3, 6, 7] if model[i].argmax() == 0]
    assert central == 1 and len(vote) >= 2 and any((model[i, 0] >= 0.5 for i in vote)) and (item.tag == 'line')
    cap = cv2.VideoCapture(item.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = round((start + offset) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    assert ok
    crops, meta = views(frame)
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(2, 3, figsize=(12, 8.2), facecolor='#f5f7f7')
    fig.suptitle(f'VF {section} · {int(start // 60):02d}:{int(start % 60):02d}–{int((start + 7) // 60):02d}:{int((start + 7) % 60):02d} · fold 2', fontsize=17, fontweight='bold', y=0.99)
    for a, k in zip(ax.flat, [0, 2, 3, 6, 7]):
        a.imshow(cv2.cvtColor(crops[k], cv2.COLOR_BGR2RGB))
        a.set_xticks([])
        a.set_yticks([])
        pred = labels[int(model[k].argmax())]
        score = 100 * model[k, 0]
        a.set_title(('CENTRAL' if k == 0 else f'RECORTE {k}') + f'\n{pred} · P(line) {score:.1f} %', fontsize=11, fontweight='bold', color='#102d40')
        for spine in a.spines.values():
            spine.set_color(colors[k])
            spine.set_linewidth(3)
    ax.flat[-1].axis('off')
    fig.text(0.5, 0.018, 'Fotograma ilustrativo. Las decisiones y puntuaciones corresponden al bloque completo de 7 segundos.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0.01, 0.06, 0.99, 0.91])
    fig.savefig(OUT / f'line_adicional_{case:02}.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    foldresc = []
    for j, m in enumerate(probs):
        vs = [i for i in [2, 3, 6, 7] if m[i].argmax() == 0]
        if m[0].argmax() == 1 and len(vs) >= 2 and any((m[i, 0] >= 0.5 for i in vs)):
            foldresc.append(j + 1)
    evidence.append({'case': case, 'section': section, 'start_s': start, 'end_s': start + 7, 'frame_time': idx / fps, 'clip_id': item.clip_id, 'fold_example': 2, 'rescued_folds': foldresc, 'predictions': {str(k): {'class': labels[int(model[k].argmax())], 'p_line': float(model[k, 0])} for k in [0, 2, 3, 6, 7]}})
(OUT / 'line_adicional_evidencia.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(evidence, indent=2))
