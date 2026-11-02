"""Recalculate metrics independently, publish comparison and archive immutable evidence."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json, time, zipfile, urllib.request, html
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def independent_metrics(y, p):
    pr, re, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    d = dict(f1_macro=float(f1_score(y, p, labels=[0, 1, 2], average='macro', zero_division=0)), accuracy=float(accuracy_score(y, p)))
    for j, k in enumerate(LABELS):
        d.update({k + '_precision': pr[j], k + '_recall': re[j], k + '_f1': f[j]})
    return d

def main():
    check()
    results = pd.read_csv(_resolve_model_artifact(HERE / 'results.csv'))
    assert len(results) == 9
    comp = load(HERE / 'paired_bootstrap.json')
    preds = {k: [] for k in ['central', 'cost_2', 'cost_4']}
    frame = None
    tc = load(HERE / 'training_complete.json')
    started = load(HERE / 'evaluation_started.json')['started_unix']
    assert tc['completed_unix'] <= started
    for seed in SEEDS:
        for variant in preds:
            out = HERE / f'seed_{seed}' / variant
            df = pd.read_csv(_resolve_model_artifact(out / 'vf_predictions.csv'))
            p = _load_model_result_array(out / 'vf_probabilities.npy')
            y = np.array([LABELS.index(k) for k in df.y_true])
            pred = p.argmax(1)
            assert len(df) == 688 and df.clip_id.is_unique
            if frame is None:
                frame = df
            else:
                assert frame.clip_id.tolist() == df.clip_id.tolist() and frame.y_true.tolist() == df.y_true.tolist()
            np.testing.assert_allclose(p, df[['p_' + k for k in LABELS]], atol=1e-07)
            np.testing.assert_array_equal(pred, [LABELS.index(k) for k in df.y_final])
            mm = independent_metrics(y, pred)
            record = results[(results.seed == seed) & (results.variant == variant)].iloc[0]
            for key, value in mm.items():
                np.testing.assert_allclose(value, record[key], atol=1e-12)
            cm = confusion_matrix(y, pred, labels=[0, 1, 2])
            saved = load(out / 'vf_metrics.json')
            for j, k in enumerate(LABELS):
                cc = dict(tp=int(cm[j, j]), fn=int(cm[j].sum() - cm[j, j]), fp=int(cm[:, j].sum() - cm[j, j]), support=int(cm[j].sum()))
                assert cc == saved['counts'][k]
                for name, value in cc.items():
                    assert record[k + '_' + name] == value
            if variant == 'central':
                np.testing.assert_allclose(p, _load_model_result_array(PARENT / f'seed_{seed}/fixed/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
                base = pred
            else:
                selected = load(out / 'selected.json')
                assert selected['saved_unix'] <= started
                assert digest(out / 'best.weights.h5') == selected['checkpoint_sha256']
                assert selected['initial_core_sha256'] == load(PARENT / f'seed_{seed}/fixed/selected.json')['initial_core_sha256']
                assert len(pd.read_csv(_resolve_model_artifact(out / 'history.csv'))) == 15
            assert record.corrected == int(((base != y) & (pred == y)).sum()) and record.broken == int(((base == y) & (pred != y)).sum())
            preds[variant].append(pred)
    sections = frame.section.to_numpy()
    bins = (frame.start_s.to_numpy() // 60).astype(int)
    blocks = [[np.where((sections == s) & (bins == b))[0] for b in sorted(set(bins[sections == s]))] for s in sorted(set(sections))]
    for variant in ['cost_2', 'cost_4']:
        rng = np.random.default_rng(20261027)
        deltas = []
        for _ in range(2000):
            ii = np.concatenate([np.concatenate([block[j] for j in rng.integers(len(block), size=len(block))]) for block in blocks])
            if any((np.sum(y[ii] == j) == 0 for j in range(3))):
                continue
            delta = []
            for j in [0, 2]:
                event = ii[y[ii] == j]
                delta.append(np.mean([np.mean(n[event] == j) - np.mean(b[event] == j) for b, n in zip(preds['central'], preds[variant])]))
            deltas.append(delta)
        for j, k in enumerate(['line', 'scrum']):
            np.testing.assert_allclose(np.quantile(np.asarray(deltas)[:, j], [0.025, 0.975]), comp[variant]['metrics'][k + '_recall']['ci95'], atol=1e-12)
    report = ROOT / 'src/web/static/reports' / HERE.name
    report.mkdir(parents=True, exist_ok=True)
    means = results.groupby('variant').mean(numeric_only=True).reindex(['central', 'cost_2', 'cost_4'])
    keys = ['line_precision', 'line_recall', 'line_f1', 'scrum_precision', 'scrum_recall', 'scrum_f1', 'f1_macro', 'accuracy']
    labels = ['Line: precision', 'Line: sensibilidad', 'Line: F1', 'Scrum: precision', 'Scrum: sensibilidad', 'Scrum: F1', 'F1 macro', 'Exactitud']
    table = pd.DataFrame({v: 100 * means.loc[v, keys].to_numpy(dtype=float) for v in means.index}, index=labels)
    count_keys = ['line_tp', 'line_fn', 'line_fp', 'scrum_tp', 'scrum_fn', 'scrum_fp', 'corrected', 'broken']
    counts = means[count_keys]
    class_changes = []
    for seed in SEEDS:
        for variant in ['cost_2', 'cost_4']:
            change = load(HERE / f'seed_{seed}' / variant / 'vf_metrics.json')['changes']['by_true_class']
            for label, values in change.items():
                class_changes.append(dict(seed=seed, variant=variant, true_class=label, **values))
    class_changes = pd.DataFrame(class_changes)
    cirows = []
    for v in ['cost_2', 'cost_4']:
        for k, value in comp[v]['metrics'].items():
            cirows.append(dict(variante=v, metrica=k, diferencia_pp=100 * value['difference'], IC95_inferior=100 * value['ci95'][0], IC95_superior=100 * value['ci95'][1]))
    fig, axes = plt.subplots(4, 1, figsize=(15, 9), constrained_layout=True)
    colors = np.array(['#19a7a0', '#dddfe4', '#ed9a36'])
    for ax, s in zip(axes, sorted(set(sections))):
        ix = np.where(sections == s)[0]
        a = np.stack([y[ix]] + [preds[v][1][ix] for v in preds])
        from matplotlib.colors import ListedColormap
        ax.imshow(a, aspect='auto', interpolation='nearest', cmap=ListedColormap(colors), vmin=0, vmax=2, extent=[0, float(frame.iloc[ix[-1]].end_s), 3.5, -0.5])
        ax.set_yticks(range(4), ['Humano', 'Central', 'Costo2', 'Costo4'])
        ax.set_title(s)
        ax.set_xlabel('Segundos')
    fig.suptitle('VF semilla42 prefijada: line verde, other gris, scrum naranja; argmax sin umbrales')
    fig.savefig(report / 'timelines.png', dpi=130)
    plt.close(fig)
    timing = load(HERE / 'timing.json')
    training_seconds = sum((r['seconds'] for r in timing['training']))
    infer_seconds = sum((r['inference_seconds'] for r in timing['inference']))
    sections_html = []
    for v in ['cost_2', 'cost_4']:
        both = all((comp[v]['metrics'][k + '_recall']['difference'] > 0 for k in ['line', 'scrum']))
        sections_html.append('<li>' + v + ': ' + ('aumentan ambas sensibilidades medias' if both else 'no aumentan ambas sensibilidades medias') + '.</li>')
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><title>Central: costo de omitir line y scrum</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#19233b}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}img{width:100%}p{line-height:1.6}.note{background:#fff0ce;padding:18px}</style><h1>Central orientado a sensibilidad: D2 / VF</h1><p>Dos costos prefijados: multiplicadores [2,1,2] y [4,1,4] para [line,other,scrum], sobre pesos inversos a frecuencia, normalizados a media uno. Central original como referencia. Misma arquitectura, inicializacion por semilla y orden de minibatches; RNN y clasificador reentrenados, ResNet congelada. 8796 ejemplos D2, 15 epocas, semillas17/42/2026. Argmax directo, sin umbrales ni ajustes de probabilidades.</p><p class="note">Evaluacion exploratoria: VF fue observado previamente y CH tiene procedencia incompleta. Dos variantes y multiples metricas; IC95 descriptivos con bloques temporales60s dentro de seccion. 688 bloques humanos (68 line,39 scrum,581 other), no eventos unicos. La media de tres semillas no es un ensamble. No se eligieron costos ni epocas usando VF.</p><h2>Porcentaje medio entre semillas</h2>'
    doc += table.to_html(float_format=lambda x: f'{x:.2f}') + '<ul>' + ''.join(sections_html) + '</ul><h2>TP, FN y FP: promedio por ejecucion</h2><p>TP: bloques verdaderos recuperados. FN: bloques de la clase omitidos. FP: bloques de otra clase predichos como la clase. No son alertas o jugadas unicas.</p>' + counts.to_html(float_format=lambda x: f'{x:.2f}')
    doc += '<h2>Cambios pareados e IC95 temporal</h2>' + pd.DataFrame(cirows).to_html(index=False, float_format=lambda x: f'{x:.2f}')
    doc += '<h2>Resultados por semilla</h2>' + results[['seed', 'variant'] + keys].to_html(index=False, float_format=lambda x: f'{100 * x:.2f}')
    doc += '<h2>TP, FN y FP por semilla</h2>' + results[['seed', 'variant'] + count_keys].to_html(index=False)
    doc += '<h2>Corregidos y rotos por clase verdadera</h2><p>Promedios por ejecucion:</p>' + class_changes.groupby(['variant', 'true_class'])[['corrected', 'broken', 'support']].mean().to_html(float_format=lambda x: f'{x:.2f}') + class_changes.to_html(index=False)
    doc += '<h2>Timeline por bloques</h2><img src="timelines.png"><h2>Tiempos</h2><p>Seis entrenamientos nuevos: %.2f s. Nueve inferencias desde caracteristicas ResNet: %.2f s. Caracteristicas reutilizadas; no es un benchmark completo desde video.</p>' % (training_seconds, infer_seconds)
    doc += '<p>Predicciones, probabilidades, pesos y protocolo conservados en ' + html.escape(str(HERE)) + '. Verificacion independiente de nueve resultados y de intervalos de sensibilidad.</p></html>'
    (report / 'index.html').write_text(doc, encoding='utf-8')
    http = {}
    for name in ['index.html', 'timelines.png']:
        url = 'http://localhost:5000/static/reports/' + HERE.name + '/' + name
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                http[name] = response.status
        except Exception as e:
            http[name] = repr(e)
    http_passed = all((v == 200 for v in http.values()))
    save(HERE / 'verification.json', dict(passed=http_passed, metrics_passed=True, prediction_sets=9, sklearn_metrics=True, confusion_counts=True, base_reproduced=True, recall_temporal_ci_independent=True, all_models_before_vf=True, http=http))
    backup = ROOT / 'model_backups' / (HERE.name + '_verified.zip')
    backup.parent.mkdir(exist_ok=True)
    names = [p for p in _model_artifact_glob(HERE, '*', recursive=True) if _resolve_model_artifact(p).is_file() and p.suffix in ['.py', '.json', '.csv', '.npy', '.h5', '.md'] and ('analysis' not in p.parts) and (p.name not in ['process.json', 'status.json', 'completion.json', 'backup_complete.json', 'pipeline_complete.json', 'run.json'])]
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in names:
            z.write(p, p.relative_to(HERE))
    with zipfile.ZipFile(backup) as z:
        assert z.testzip() is None
        for p in names:
            assert hashlib.sha256(z.read(str(p.relative_to(HERE)).replace('\\', '/'))).hexdigest() == digest(p)
    save(HERE / 'backup_complete.json', dict(passed=True, path=str(backup), sha256=digest(backup), entries=len(names)))
    if not http_passed:
        raise RuntimeError('Models backed up and metrics verified; HTTP report unavailable: ' + str(http))
    save(HERE / 'completion.json', dict(complete=True, verified=True, report=str(report / 'index.html'), training_seconds=training_seconds, inference_seconds=infer_seconds))
    print('COMPLETE', means[keys].to_json(), flush=True)
if __name__ == '__main__':
    with lock('publication.lock'):
        main()
