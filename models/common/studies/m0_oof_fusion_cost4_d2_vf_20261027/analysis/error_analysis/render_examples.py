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
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
OUT = experiment_directory(__file__)
P = ROOT / 'data/central_preserving_d2_vf_20261026'
sys.path.insert(0, str(source_directory(P)))
from common import views, read_samples, load
df = pd.read_csv(_resolve_model_artifact(OUT / 'todos_bloques_y_semillas.csv'))
rows = {r['clip_id']: r for r in load(P / 'manifest.json')}
cases = [('line_adicional', 'S01', 1085), ('falsa_alarma', 'S01', 168), ('scrum_sin_voto', 'S03', 1127)]
for name, section, start in cases:
    r = df[(df.seed == 42) & (df.section == section) & (df.start_s == start)].iloc[0]
    data = rows[r.clip_id]
    samples = list(read_samples(data))
    n, frame = samples[len(samples) // 2]
    crops, _ = views(frame)
    fig = plt.figure(figsize=(12, 11))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.25, 1, 1, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(frame[:, :, ::-1])
    ax.axis('off')
    ax.set_title(f'{section} · {int(start // 60):02}:{int(start % 60):02}–{int((start + 7) // 60):02}:{int((start + 7) % 60):02} · cuadro representativo')
    for j in range(9):
        ax = fig.add_subplot(grid[1 + j // 3, j % 3])
        ax.imshow(crops[j][:, :, ::-1])
        ax.set_title(f"{('Central' if j == 0 else 'Vista ' + str(j))}: {r[f'vista_{j}']}")
        ax.axis('off')
    fig.suptitle(f'Anotación: {r.human} · combinación: {r.fusion} · regla votos: {r.regla_votos_evento}\nVotos: line {r.line_votes}, scrum {r.scrum_votes}, other {r.other_votes} · semilla 42', fontsize=13)
    fig.text(0.02, 0.012, 'La etiqueta y las predicciones corresponden a un bloque de 7 s; un cuadro no demuestra por qué votó cada vista.', fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT / (name + '.png'), dpi=105)
    plt.close(fig)
print('saved', len(cases), 'actual video examples')
