"""Show actual nine crops and decisions for selected VF errors; no causal claim."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
R = ROOT / 'data/m0_vote_event_cost4_vf_20261027'
P = ROOT / 'data/central_preserving_d2_vf_20261026'
sys.path.insert(0, str(source_directory(P)))
from common import views, read_samples, load

def main():
    df = pd.read_csv(_resolve_model_artifact(R / 'predictions.csv'))
    df = df[df.seed == 42]
    manifest = {r['clip_id']: r for r in load(P / 'manifest.json')}
    cases = [('line_como_scrum', df[(df.y_true == 'line') & (df.priority == 'scrum')]), ('scrum_como_line', df[(df.y_true == 'scrum') & (df.priority == 'line') & (df.central == 'other')]), ('otro_como_line', df[(df.y_true == 'other') & (df.priority == 'line') & (df.central == 'other')]), ('otro_como_scrum', df[(df.y_true == 'other') & (df.priority == 'scrum') & (df.central == 'other')])]
    result = []
    out = R / 'analysis'
    for name, candidates in cases:
        assert len(candidates)
        r = candidates.iloc[0]
        item = manifest[r.clip_id]
        samples = list(read_samples(item))
        index, frame = samples[len(samples) // 2]
        crops, geometry = views(frame)
        fig = plt.figure(figsize=(13, 12))
        grid = fig.add_gridspec(4, 3, height_ratios=[1.35, 1, 1, 1])
        ax = fig.add_subplot(grid[0, :])
        ax.imshow(frame[:, :, ::-1])
        ax.axis('off')
        ax.set_title(f"Original: {r.section} · {r.start_s:.0f}–{r.start_s + item['frames'] / 25:.0f} s · muestra central del clip")
        for i in range(9):
            ax = fig.add_subplot(grid[1 + i // 3, i % 3])
            probs = np.array([r[f'v{i}_{k}'] for k in ['line', 'other', 'scrum']])
            label = ['line', 'other', 'scrum'][probs.argmax()]
            ax.imshow(crops[i][:, :, ::-1])
            ax.set_title(f"{('Central' if i == 0 else 'Vista ' + str(i))}: {label} ({100 * probs.max():.0f}%)", fontsize=10)
            ax.axis('off')
        fig.suptitle(f'Etiqueta humana: {r.y_true} · central: {r.central} · votos: {r.priority}\nVotos line {r.line_votes} / scrum {r.scrum_votes} / other {r.other_votes}', fontsize=14)
        fig.text(0.05, 0.01, 'Semilla 42. Un cuadro ilustra un bloque temporal completo; la etiqueta y la predicción corresponden al bloque entero.', fontsize=9)
        fig.tight_layout(rect=[0, 0.035, 1, 0.94])
        file = out / (name + '.png')
        fig.savefig(file, dpi=115)
        plt.close(fig)
        result.append(dict(name=name, clip_id=r.clip_id, section=r.section, start_s=float(r.start_s), end_s=float(r.start_s + item['frames'] / 25), source_frame=int(index), human=r.y_true, central=r.central, decision=r.priority, votes=dict(line=int(r.line_votes), scrum=int(r.scrum_votes), other=int(r.other_votes)), file=str(file)))
    (out / 'examples.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(result, ensure_ascii=True))
if __name__ == '__main__':
    main()
