"""Early validation of the definitive seed17 initial heads; no test tuning."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import time, json
import numpy as np, pandas as pd
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.experiment import OUT, LABELS, guard, manifest, initial_dir, predict, digest
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.preflight import Peak, write_json

def run(seed=17):
    import keras
    from models.common.comparison.metrics import block_metrics
    guard()
    target = OUT / 'early_check.json'
    if _resolve_model_artifact(target).exists():
        return
    split = pd.read_csv(_resolve_model_artifact(OUT / 'initial_split.csv'))
    m = manifest()
    ch = m[m.domain == 'CH'].set_index('video_name')
    rows = ch.loc[split.video_name].reset_index()
    rows['initial_partition'] = split.initial_partition.to_numpy()
    val = rows[rows.initial_partition == 'validation'].copy()
    result = {}
    tick = time.perf_counter()
    with Peak() as peak:
        for size in (224, 320):
            folder = initial_dir(size, seed)
            meta = json.loads(_resolve_model_artifact(folder / 'complete.json').read_text())
            assert digest(folder / 'best.keras') == meta['checkpoint_sha256']
            model = keras.models.load_model(_resolve_model_artifact(folder / 'best.keras'), compile=False)
            p = predict(model, val, size, seed)
            assert len(p) == 360 and np.isfinite(p).all() and np.allclose(p.sum(1), 1)
            table = val[['block_id', 'video_name', 'label', 'cache_index']].copy()
            for i, label in enumerate(LABELS):
                table['p_' + label] = p[:, i]
            table.to_csv(OUT / f'early_check_R{size}_S{seed}_validation.csv', index=False)
            result[str(size)] = dict(metrics=block_metrics(np.asarray([LABELS.index(v) for v in val.label]), p), checkpoint_sha256=meta['checkpoint_sha256'])
            del model
    result = dict(status='complete', seed=seed, conditions=result, elapsed_s=time.perf_counter() - tick, peak_rss_bytes=peak.value, scope='Initial clip validation only. Same360 validation rows used by early stopping. Not temporal match evaluation, not external confirmation.65 shared initial train/validation names. No protocol changes based on these scores.', reused_in_full_run=True)
    write_json(target, result)
    lines = ['Prueba inicial definitiva, semilla17', 'ValidaciÃ³n de clips del entrenamiento inicial. No es F1 temporal en partidos.', 'Recorte | accuracy | F1 macro (3 clases) | F1 line | F1 scrum']
    for size, r in result['conditions'].items():
        b = r['metrics']
        lines.append(f"{size} | {b['accuracy']:.4f} | {b['f1_macro']:.4f} | {b['per_class']['line']['f1']:.4f} | {b['per_class']['scrum']['f1']:.4f}")
    lines += ['Estos checkpoints y caracterÃ­sticas se usan en el experimento completo. No se repite este entrenamiento.', 'La particiÃ³n tiene65 nombres compartidos entre entrenamiento y validaciÃ³n. Los resultados son diagnÃ³sticos y no una prueba independiente.']
    (OUT / 'early_check.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('EARLY_CHECK_COMPLETE', flush=True)
if __name__ == '__main__':
    run()
