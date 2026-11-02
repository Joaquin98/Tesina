"""Real VF examples of central errors corrected and central successes broken."""
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
import sys, json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = experiment_directory(__file__).parents[0]
sys.path.insert(0, str(source_directory(R)))
from common import read_samples, views, load

def main():
    assert load(R / 'completion.json')['verified']
    report = R.parents[1] / 'src/web/static/reports' / R.name
    out = report / 'examples'
    out.mkdir(exist_ok=True)
    rows = {r['clip_id']: r for r in load(R / 'manifest.json')}
    base = pd.read_csv(_resolve_model_artifact(R / 'seed_42/central/vf_predictions.csv'))
    records = []
    for strategy in ['residual', 'late_mix', 'protected_residual']:
        frame = pd.read_csv(_resolve_model_artifact(R / f'seed_42/{strategy}/vf_predictions.csv'))
        a = _load_model_result_array(R / f'seed_42/{strategy}/vf_attention.npy')
        assert list(frame.clip_id) == list(base.clip_id)
        for kind in ['corrected', 'broken']:
            bc = base.y_final.eq(base.y_true)
            nc = frame.y_final.eq(frame.y_true)
            candidates = np.flatnonzero((~bc & nc if kind == 'corrected' else bc & ~nc) & frame.y_true.isin(['line', 'scrum']))
            if not len(candidates) and kind == 'corrected':
                candidates = np.flatnonzero(~bc & nc)
            if not len(candidates):
                continue
            i = int(candidates[0])
            row = frame.iloc[i]
            source = rows[row.clip_id]
            sampled = list(read_samples(source))
            frame_index, pixels = sampled[len(sampled) // 2]
            crops, geometry = views(pixels)
            top = (np.argsort(a[i, 1:])[::-1][:2] + 1).tolist()
            chosen = [0] + top
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), gridspec_kw={'height_ratios': [1.2, 1]})
            for ax in axes[0]:
                ax.axis('off')
            axes[0, 0].imshow(pixels[:, :, ::-1])
            axes[0, 0].set_title('Fotograma original', fontsize=10)
            axes[0, 1].text(0, 0.9, f'{strategy} · {kind}\n{row.section}: {row.start_s:.1f}–{row.end_s:.1f}s\nEtiqueta humana: {row.y_true}\nCentral: {base.iloc[i].y_final}\nNuevo: {row.y_final}', va='top', fontsize=12)
            labels = ['line', 'other', 'scrum']
            bp = [base.iloc[i]['p_' + k] for k in labels]
            npred = [row['p_' + k] for k in labels]
            axes[0, 2].text(0, 0.9, 'Probabilidades central → nuevo\n' + '\n'.join((f'{k}: {100 * x:.1f}% → {100 * y:.1f}%' for k, x, y in zip(labels, bp, npred))), va='top', fontsize=11)
            for ax, v in zip(axes[1], chosen):
                ax.imshow(crops[v][:, :, ::-1])
                ax.set_title(('Central fijo' if v == 0 else f'Contexto{v}') + f' · peso {100 * a[i, v]:.1f}%')
                ax.axis('off')
            fig.suptitle('Semilla42 · ejemplos seleccionados, no muestra aleatoria', fontsize=14)
            fig.text(0.03, 0.02, 'Peso residual: sólo atención a contextos; central entra por vía fija aunque marque0%. Mezcla: coeficiente de probabilidad.\nUn fotograma ilustra un clip completo; los pesos no demuestran causalidad.', fontsize=9)
            fig.tight_layout(rect=[0, 0.09, 1, 0.95])
            filename = f'{strategy}_{kind}_{i}.png'
            fig.savefig(out / filename, dpi=125)
            plt.close(fig)
            records.append(dict(strategy=strategy, kind=kind, test_index=i, clip_id=row.clip_id, human=row.y_true, central=base.iloc[i].y_final, new=row.y_final, frame_index=frame_index, views=chosen, weights=a[i].tolist(), file=filename))
    (out / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf8')
    section = '<section id="error-examples"><h2>Errores corregidos y aciertos rotos del central</h2><p>Ejemplos semilla42, priorizando bloques line/scrum. Si no se recupera una jugada real, se muestra una falsa alarma corregida en un bloque other. No son una muestra aleatoria. Los pesos no son evidencia causal.</p>'
    for r in records:
        section += f"""<h3>{r['strategy']} · {r['kind']}</h3><img style="width:100%" src="examples/{r['file']}">"""
    section += '</section>'
    path = report / 'index.html'
    page = _resolve_model_artifact(path).read_text(encoding='utf8')
    if 'id="error-examples"' in page:
        page = re.sub('<section id="error-examples">.*?</section>', lambda _: section, page, flags=re.S)
    else:
        page = page.replace('</html>', section + '</html>')
    path.write_text(page, encoding='utf8')
    print(json.dumps(records, indent=2))
if __name__ == '__main__':
    main()
