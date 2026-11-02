"""Audit saved full-density predictions of the initial curated-clip CNN-RNN.

Run from any directory with Python 3. No model execution or third-party modules.
All generated outputs stay beside this script. Original artifacts are read-only.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
OUT = experiment_directory(__file__)
ROOT = OUT.parents[1]
BASE = ROOT / 'models/M0'
PRED = BASE / 'results/best_model_sparse_frames/predictions.csv'
LABELS = ['other', 'scrum', 'line']
DISPLAY = {'other': 'otro', 'scrum': 'scrum', 'line': 'lineout'}
CATEGORIES = ['ataque', 'defensa', 'juego', 'maul', 'penales', 'pie', 'palos', 'recepciones', 'salidas', 'no_determinado']
CAT_DISPLAY = {x: x for x in CATEGORIES}
CAT_DISPLAY.update(juego='juego (sin fase específica)', pie='pie (origen)', palos='palos (origen)', no_determinado='no determinado')

def read(path):
    with _resolve_model_artifact(study_csv_path(Path(path))).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with _resolve_model_artifact(study_csv_path(OUT / name)).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha(path):
    h = hashlib.sha256()
    with _resolve_model_artifact(study_csv_path(Path(path))).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def category(name):
    tokens = name.lower().split('_')
    found = [x for x in ['ataque', 'defensa', 'juego', 'maul', 'penales', 'pie', 'palos', 'recepcion', 'recepciones', 'salida', 'salidas'] if x in tokens]
    assert len(found) <= 1, (name, found)
    if not found:
        return ('no_determinado', '')
    token = found[0]
    return ({'recepcion': 'recepciones', 'salida': 'salidas'}.get(token, token), token)

def pct(n, d):
    return round(100 * n / d, 6) if d else None

def table(rows):
    lines = ['| Clase real / categoría de origen | n | → otro | → scrum | → lineout | Errores/n | % de los 154 errores |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['category_display']} | {r['n']} | {r['pred_other']} | {r['pred_scrum']} | {r['pred_line']} | {r['errors']}/{r['n']} ({r['error_rate_pct']:.2f}%) | {r['share_all_errors_pct']:.2f}% |")
    return '\n'.join(lines)

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    all_predictions = read(PRED)
    predictions = [dict(r, prediction_csv_line=i + 2) for i, r in enumerate(all_predictions) if r['run'] == 'full_density']
    test = read(CURRENT_DATASET / 'test.csv')
    original_test = read(CURRENT_DATASET / 'test.csv')
    train = read(CURRENT_DATASET / 'train.csv')
    original_train = read(CURRENT_DATASET / 'train.csv')
    assert test == original_test
    assert len(predictions) == len(test) == 817
    assert len({r['video_name'].lower() for r in predictions}) == 817
    assert [(r['video_name'], r['true_label']) for r in predictions] == [(r['video_name'], r['tag']) for r in test]
    assert not {r['video_name'].lower() for r in test} & {r['video_name'].lower() for r in train}
    assert Counter((r['true_label'] for r in predictions)) == Counter(other=544, line=183, scrum=90)
    cm = Counter(((r['true_label'], r['predicted_label']) for r in predictions))
    assert [[cm[a, b] for b in LABELS] for a in LABELS] == [[431, 27, 86], [11, 76, 3], [25, 2, 156]]
    metrics = {}
    f1s = []
    weighted_f1 = 0
    for label in LABELS:
        support = sum((cm[label, b] for b in LABELS))
        predicted = sum((cm[a, label] for a in LABELS))
        f1 = 2 * cm[label, label] / (support + predicted)
        f1s.append(f1)
        weighted_f1 += support * f1 / 817
        metrics[label] = dict(support=support, predicted=predicted, correct=cm[label, label], recall=cm[label, label] / support, precision=cm[label, label] / predicted, f1=f1)
    accuracy = sum((cm[a, a] for a in LABELS)) / 817
    macro_f1 = sum(f1s) / 3
    sparse_summary = read(BASE / 'results/best_model_sparse_frames/summary.csv')[0]
    training_result = read(BASE / 'results/best_model/best_model_results.csv')[0]
    for key, value in [('accuracy', accuracy), ('f1_macro', macro_f1), ('f1_weighted', weighted_f1)]:
        assert abs(value - float(sparse_summary[key])) < 1e-06
    assert abs(accuracy - float(training_result['test_accuracy'])) < 2e-06
    assert abs(macro_f1 - float(training_result['f1_macro'])) < 5e-05
    for label in LABELS:
        assert abs(metrics[label]['recall'] - float(training_result['recall_' + label])) < 5e-05
    checkpoint = BASE / 'checkpoints/best_model/best_model.keras'
    metadata = json.loads(_resolve_model_artifact(study_csv_path(BASE / 'results/best_model_sparse_frames/metadata.json')).read_text())
    assert Path(metadata['checkpoint_path']).resolve() == checkpoint.resolve()
    assert Path(metadata['test_csv']).resolve() == (CURRENT_DATASET / 'test.csv').resolve()
    assert metadata['label_vocab'] == ['line', 'other', 'scrum']
    with zipfile.ZipFile(checkpoint) as archive:
        keras_metadata = json.loads(archive.read('metadata.json'))
        config = json.loads(archive.read('config.json'))
    recurrent = [x for x in config['config']['layers'] if x['class_name'] == 'Bidirectional']
    assert [x['config']['layer']['class_name'] for x in recurrent] == ['GRU', 'GRU']
    assert [x['config']['layer']['config']['units'] for x in recurrent] == [64, 32]
    trial_name = 'resnet50__gru__u64__bi__drop3__d64__lr1e-4__bs16'
    trial = next((r for r in read(BASE / 'results/grid_search_results.csv') if r['trial_name'] == trial_name))
    assert float(trial['val_f1_macro']) == 0.7973
    regeneration = {r['output_name'].lower(): r for r in read(ROOT / 'data/7seg_exact_7s_half_center224_20260716/regeneration_manifest.csv')}
    master = {r['video_name'].lower(): r for r in read(ROOT / 'data/final_model_comparison_20260827/outputs/master_examples.csv') if r['domain'] == 'CH'}
    transition_names = {'Todo_R1_maul_001_cut.mp4', 'Todo_F11_maul_004_cut.mp4', 'Todo_F13_maul_001_cut.mp4'}
    person_names = {'Todo_F7_Fede_007_cut.mp4', 'Todo_F7_Julian_002_cut.mp4', 'Todo_F7_Leandro_000_cut.mp4', 'Todo_F7_mateo_001_cut.mp4'}
    visual = []
    details = []
    for i, r in enumerate(predictions):
        name = r['video_name']
        cat, token = category(name) if r['true_label'] == 'other' else (r['true_label'], r['true_label'])
        source = ROOT / 'data/7seg unique' / name
        raw_source = ROOT / 'data/7Seg' / name.removeprefix('Todo_')
        processed = ROOT / 'data/videos/clips_224' / name
        assert _resolve_model_artifact(study_csv_path(source)).exists() and _resolve_model_artifact(study_csv_path(raw_source)).exists() and _resolve_model_artifact(study_csv_path(processed)).exists()
        assert master[name.lower()]['label'] == r['true_label']
        assert Path(regeneration[name.lower()]['source_file']).resolve() == raw_source.resolve()
        sheet = OUT / 'visual_checks' / (Path(name).stem + '.png')
        reviewed = _resolve_model_artifact(study_csv_path(sheet)).exists()
        observation = ''
        if name in transition_names:
            observation = 'Salto/recepción de lineout al comienzo, seguido de agrupamiento compatible con maul. Visible también en la versión procesada 224x224.'
        elif reviewed and cat == 'maul':
            observation = 'Agrupamiento de contacto compatible con el origen maul. No se confirma su inicio ni una fase previa mediante estos fotogramas.'
            if name == 'Todo_f4_maul_005_cut.mp4':
                observation = 'Contacto junto al suelo, interrupción y cambio de secuencia. La fase maul no queda confirmada en los fotogramas muestreados.'
            if name == 'Todo_F18_maul_000_cut.mp4':
                observation += ' Hay un cambio de secuencia en el montaje.'
        elif name in person_names:
            observation = 'Juego general, sin significado de fase recuperable del nombre personal. Se conserva no determinado.'
        status = 'sampled_video_frames' if reviewed else 'not_reviewed'
        row = dict(video_name=name, true_label=r['true_label'], predicted_label=r['predicted_label'], is_error=int(r['true_label'] != r['predicted_label']), confidence=r['confidence'], category=cat, category_token=token, category_basis='original_filename_token' if token else 'undetermined_filename', source_group=master[name.lower()]['source_group'], split_csv_line=i + 2, prediction_csv_line=r['prediction_csv_line'], run='full_density', selected_frames=r['selected_frames'], source_video=str(source), raw_source_video=str(raw_source), processed_video=str(processed), master_block_id=master[name.lower()]['block_id'], regeneration_metadata_only='not_used_for_evaluation', visual_status=status, visual_sheet=str(sheet) if reviewed else '', lineout_to_contact_observed=int(name in transition_names) if reviewed else '', visual_observation=observation)
        details.append(row)
        if reviewed:
            visual.append({k: row[k] for k in ['video_name', 'true_label', 'predicted_label', 'category', 'source_video', 'processed_video', 'visual_status', 'visual_sheet', 'lineout_to_contact_observed', 'visual_observation']})
    write('per_clip_audit.csv', details)
    write('visual_review.csv', visual)
    write('confusion_matrix.csv', [dict(true_label=a, n=sum((cm[a, b] for b in LABELS)), **{'pred_' + b: cm[a, b] for b in LABELS}) for a in LABELS])
    summary = []
    for cat in ['scrum', 'line'] + CATEGORIES:
        rows = [r for r in details if r['category'] == cat]
        n = len(rows)
        counts = Counter((r['predicted_label'] for r in rows))
        errors = sum((r['is_error'] for r in rows))
        summary.append(dict(category=cat, category_display=DISPLAY.get(cat, 'otro: ' + CAT_DISPLAY.get(cat, cat)), n=n, pred_other=counts['other'], pred_scrum=counts['scrum'], pred_line=counts['line'], errors=errors, error_rate_pct=pct(errors, n), share_all_errors_pct=pct(errors, 154), share_other_errors_pct=pct(errors, 113) if cat in CATEGORIES else None, rate_pred_scrum_pct=pct(counts['scrum'], n), rate_pred_line_pct=pct(counts['line'], n)))
    assert sum((r['n'] for r in summary)) == 817
    assert sum((r['errors'] for r in summary)) == 154
    write('table_by_origin.csv', summary)
    directions = [dict(true_label=a, predicted_label=b, count=cm[a, b], true_class_n=metrics[a]['support'], rate_within_true_class_pct=pct(cm[a, b], metrics[a]['support']), share_all_errors_pct=pct(cm[a, b], 154), share_errors_true_class_pct=pct(cm[a, b], metrics[a]['support'] - cm[a, a])) for a in LABELS for b in LABELS if a != b]
    write('error_directions.csv', sorted(directions, key=lambda x: -x['count']))
    split_at = math.floor(len(train) * 0.8)
    coverage = []
    for cat in CATEGORIES:
        sets = [original_train + original_test, original_train, train, train[:split_at], train[split_at:], test]
        counts = []
        for rows in sets:
            counts.append(len({r['video_name'].lower() for r in rows if r['tag'] == 'other' and category(r['video_name'])[0] == cat}))
        coverage.append(dict(category=cat, **dict(zip(['corpus_unique', 'original_train_unique', 'balanced_train_file_unique', 'optimization_subset_unique', 'validation_subset_unique', 'test_unique'], counts))))
    assert next((r for r in coverage if r['category'] == 'maul'))['optimization_subset_unique'] == 11
    write('other_coverage.csv', coverage)
    write('maul_training_clips.csv', [dict(csv_line=i + 2, video_name=r['video_name'], use='optimization' if i < split_at else 'validation') for i, r in enumerate(train) if category(r['video_name'])[0] == 'maul'])
    inputs = [PRED, CURRENT_DATASET / 'test.csv', CURRENT_DATASET / 'test.csv', CURRENT_DATASET / 'train.csv', CURRENT_DATASET / 'train.csv', checkpoint, BASE / 'results/best_model/best_model_results.csv', BASE / 'results/best_model/classification_report.txt', BASE / 'results/best_model/confusion_matrix_raw.png', BASE / 'results/best_model_sparse_frames/metadata.json', BASE / 'results/best_model_sparse_frames/summary.csv', BASE / 'results/grid_search_results.csv', _resolve_model_artifact(study_csv_path(BASE / 'train_best.py')), _resolve_model_artifact(study_csv_path(BASE / 'evaluate_best_model_sparse_frames.py')), _resolve_model_artifact(study_csv_path(BASE / 'config.py')), _resolve_model_artifact(study_csv_path(BASE / 'create_train_test_csv.py')), _resolve_model_artifact(study_csv_path(ROOT / 'scripts/data_prep/create_balanced_splits.py')), ROOT / 'data/7seg_exact_7s_half_center224_20260716/regeneration_manifest.csv', ROOT / 'data/final_model_comparison_20260827/outputs/master_examples.csv', Path(__file__)]
    for row in visual:
        inputs.extend([Path(row['source_video']), Path(row['visual_sheet'])])
        processed_sheet = OUT / 'visual_checks' / (Path(row['video_name']).stem + '_model_input.png')
        if _resolve_model_artifact(study_csv_path(processed_sheet)).exists():
            inputs.extend([Path(row['processed_video']), processed_sheet])
    write('input_sha256.csv', [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in inputs])
    result = dict(analysis_date='2026-10-17', scope='initial_curated_clips_not_M0_M3', inference_executed=False, prediction_run='full_density', samples=817, unique_clips=817, correct=663, errors=154, accuracy=accuracy, f1_macro=macro_f1, f1_weighted=weighted_f1, labels=metrics, checkpoint=str(checkpoint), checkpoint_keras_metadata=keras_metadata, saved_prediction_metadata=metadata, selected_grid_trial=trial, discarded_prediction_rows=len(all_predictions) - 817, training_rows=len(train), training_unique_clips=len({r['video_name'].lower() for r in train}), train_test_filename_overlap=0, visual_reviewed_clips=len(visual), visual_reviewed_maul_clips=sum((r['category'] == 'maul' for r in visual)), lineout_transition_examples=sorted(transition_names), caveat='Categories are filename provenance, not exhaustive visual ground truth; no ruck-level rate can be estimated.')
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = 'ANÁLISIS DE ERRORES DEL CNN-RNN INICIAL SOBRE CLIPS CURADOS\nFecha: 2026-10-17.\n\nIDENTIFICACIÓN Y VERIFICACIÓN\nCheckpoint: models/M0/checkpoints/best_model/best_model.keras.\nEs la ejecución de train_best.py guardada el 24/03/2026, con dos BiGRU de\n64 y 32 unidades, dropout 0.3 y capa densa de 64. Usa características ResNet-50\nde 2048 componentes, secuencias de hasta 200 pasos y máscaras de pasos válidos.\nbest_model_results.csv registra 33 épocas, exactitud 0.811505 y F1 macro 0.7870.\n\nSe recuperaron predicciones individuales en results/best_model_sparse_frames/\npredictions.csv, exclusivamente run=full_density. Los metadatos del 18/06/2026\nidentifican explícitamente ese checkpoint y balanced_test.csv. La matriz coincide\ncelda a celda con confusion_matrix_raw.png de la ejecución original, inspeccionada\nvisualmente. Las etiquetas, el orden y los nombres coinciden con las 817 filas de\nbalanced_test.csv, idéntico a test.csv. No hay nombres repetidos en prueba ni\nsolapamiento de nombres con el archivo de entrenamiento. Se excluyeron todas las\nvariantes sparse del CSV. No se ejecutó inferencia, reentrenamiento, bootstrap ni\nrepartición de datos: las predicciones necesarias ya estaban conservadas.\n\nEl ensayo de búsqueda resnet50__gru__u64__bi__drop3__d64__lr1e-4__bs16 es distinto:\ngrid_search_results.csv registra 18 épocas, exactitud 0.812729 y val_f1_macro\n0.7973 (nombre histórico de la columna). No se mezclan sus resultados con el\ncheckpoint de referencia. Tampoco se usa el posterior ensayo de búsqueda 0.8027.\nLa asociación histórica se verifica por metadatos, arquitectura, partición y\nmatriz/métricas coincidentes. input_sha256.csv fija las huellas actuales de las\nfuentes; el archivo histórico de metadatos no guardaba una huella del checkpoint.\n\nRESULTADOS GENERALES\n663/817 correctos, 154/817 errores (18.85%). Exactitud 0.811505508,\nF1 macro 0.787026767 y F1 ponderado 0.816868 aproximadamente.\nLa mayor dirección de error es otro→lineout: 86/544 (15.81% de otro),\n86/154 (55.84% de todos los errores) y 86/113 (76.11% de los errores de otro).\nLas otras direcciones son otro→scrum 27/544 (4.96%), lineout→otro 25/183\n(13.66%), scrum→otro 11/90 (12.22%), scrum→lineout 3/90 (3.33%) y\nlineout→scrum 2/183 (1.09%). Las últimas dos suman 5/273 (1.83% de las\njugadas objetivo), 5/154 (3.25% de los errores). Los errores de jugadas objetivo\nhacia otro suman 36/273 (13.19%) y 36/154 (23.38% de los errores).\nOtro aporta 113/154 errores (73.38%), lineout 27/154 (17.53%) y scrum\n14/154 (9.09%). Las tasas por clase son 20.77%, 14.75% y 15.56%, respectivamente.\n\nDe los 245 clips predichos lineout, 86 son otro y 3 scrum: precisión 156/245\n(63.67%). De los 105 predichos scrum, 27 son otro y 2 lineout: precisión\n76/105 (72.38%). Los denominadores de precisión no son los de sensibilidad.\n\nTABLA POR CLASE REAL Y ORIGEN DE OTRO\nLas filas son disjuntas y suman 817. Las tres columnas de predicción son cantidades.\nEn cada fila, errores/n es la tasa dentro de ese origen, mientras que la última\ncolumna mide su aporte al total de 154 errores. table_by_origin.csv contiene además\nel porcentaje entre los 113 errores de otro y las tasas específicas hacia cada clase.\n\n' + table(summary) + '\n\nINTERPRETACIÓN DE LOS ORÍGENES\nLa etiqueta real de tres clases procede del archivo de prueba. Los subtipos de\notro se recuperan de tokens exactos en el nombre original, sin usar la predicción\npara decidir el subtipo. Se unen recepcion/recepciones y salida/salidas.\nAtaque, defensa y juego son rótulos amplios, no fases homogéneas ni sinónimos de\nruck. Pie y palos conservan su rótulo de origen sin atribuirles un tipo de patada\nmás preciso. Seis nombres personales permanecen no determinados. No se ha\nconvertido la revisión parcial de video en una nueva anotación exhaustiva.\n\nLa confusión de otro con scrum aparece principalmente en ataque (9/27 falsos\npositivos procedentes de otro, 33.33%), juego (5/27, 18.52%), penales (4/27,\n14.81%) y maul (3/27, 11.11%), además de defensa (3), pie (2) y palos (1).\nPara otro→lineout, ataque aporta 23/86 (26.74%), juego 20/86 (23.26%), defensa\n11/86 (12.79%), maul 8/86 (9.30%) y salidas 8/86 (9.30%), además de penales\n(6), pie (5), recepciones (3) y palos (2).\nLa categoría ataque aporta la mayor cantidad total de errores de otro: 32/113\n(28.32%), pero su tasa es 32/139 (23.02%). La mayor tasa es maul: 11/19\n(57.89%), repartidos en 8/19 hacia lineout (42.11%) y 3/19 hacia scrum\n(15.79%). Son solo 19 clips. No se interpreta la diferencia como una estimación\nprecisa poblacional o una comparación estadística concluyente. Maul representa\n11/113 (9.73%) de los errores de otro y 11/154 (7.14%) del total.\nEntre categorías de origen, maul→lineout tiene la mayor tasa direccional (8/19).\nEntre las tres clases principales, otro→lineout tiene también la mayor tasa\ndireccional (86/544), aunque son agrupaciones y denominadores diferentes.\n\nVERIFICACIÓN EN VIDEO Y TRANSICIONES\nSe extrajeron y revisaron secuencias de hasta seis fotogramas a 1 fps de los\n19 clips de prueba con maul en el nombre y de cuatro nombres personales. Las\nplanchas están en visual_checks y las observaciones por caso en visual_review.csv.\nEs inspección visual de fotogramas de video, no reproducción íntegra ni revisión\nde un especialista. Los paneles negros al final de clips cortos son relleno.\nEn 18 de los 19 maul se ven agrupamientos compatibles con ese origen. En\nTodo_f4_maul_005_cut se ve contacto bajo, interrupción y cambio de secuencia:\nno se confirma visualmente maul en la muestra. Se conserva su origen por nombre.\n\nEn Todo_R1_maul_001_cut, Todo_F11_maul_004_cut y Todo_F13_maul_001_cut se observa\nun salto/recepción de lineout seguido de agrupamiento compatible con maul. Los\ntres están etiquetados other y se predicen line. Esto representa al menos 3/8\n(37.50%) de los maul de origen predichos lineout. La transición también es visible\nen las planchas extraídas de los videos procesados 224x224 que acompañan a las\ncaracterísticas del modelo. Los otros cinco casos no permiten confirmar mediante\nesa muestra un lineout previo. No se los cuenta como ausencia de lineout.\nEl hallazgo apoya la existencia de clips que cruzan ambas fases, pero no prueba\nqué señales usa internamente la red ni que todos los errores tengan esa causa.\nLas etiquetas originales se conservan para reproducir exactamente la evaluación.\n\nCOBERTURA DE OTRO Y LÍMITES DE LAS HIPÓTESIS\nEl corpus histórico tiene 2716 otro y 80 nombres maul. Son datos de contexto,\nno denominadores de prueba. De los 80, 61 están en el entrenamiento original\nsin balancear y 19 en prueba. El archivo balanceado realmente suministrado al\nentrenamiento/validación contiene 600 otro únicos y solo 14 maul únicos.\ntrain_best.py usa validation_split=0.2: de esos 14, los primeros 11 pertenecen\nal subconjunto de ajuste de pesos y los últimos 3 al de validación. Este cálculo\nreconstruye el corte documentado por Keras, antes de barajar y sujeto a que se\ncargaron todas las características. El programa original omite caches inválidos:\nno se conserva un manifiesto por fila de aquel ajuste que permita probar de forma\nindependiente la ausencia de omisiones históricas. Por eso 14 es la cobertura\ncierta del archivo y 11/3 la cobertura reconstruida bajo carga completa.\nother_coverage.csv desglosa estas cantidades para todos los orígenes.\nEl archivo de entrenamiento tiene 1800 filas, pero solo 1561 nombres únicos:\n600 lineout, 600 otro y 361 scrum. Las repeticiones de scrum no se cuentan como\nclips nuevos. Aquí único significa nombre de clip único, no jugada física única:\nno se eliminaron escenas relacionadas o temporalmente cercanas dentro de la\npartición original. No se toman las 61 maul del entrenamiento original como exposición\ndel modelo de referencia ni se extrapola la proporción corpus→test.\n\nNo hay token ruck entre los nombres de esta prueba ni en los nombres históricos\ndel corpus cotejado. Eso no demuestra ausencia visual de rucks: pueden estar\ndentro de ataque, defensa, juego u otros rótulos. No se puede estimar una tasa\nruck→scrum con esta anotación. Haría falta revisar y anotar por fase los videos,\nincluidos los aciertos, para disponer de su denominador. La cantidad global de\notro no prueba cobertura suficiente de cada fase. Los 14 maul de origen en el\narchivo de entrenamiento y la tasa 11/19 en prueba justifican estudiar esa\ncobertura, pero no establecen por sí solos causalidad ni el tamaño necesario.\n\nTRAZABILIDAD Y REUTILIZACIÓN\nper_clip_audit.csv: 817 filas, línea del archivo de partición y del CSV de\npredicciones, etiqueta real/predicha, confianza, pasos válidos, origen recuperado,\nrutas de videos, vínculo al bloque histórico y estado de revisión visual.\nconfusion_matrix.csv, error_directions.csv y table_by_origin.csv: tablas calculadas.\ntable_for_thesis.tex: fragmento de tabla para futura incorporación, sin incluirlo\nen main.tex. summary.json: métricas exactas y prueba de identidad de artefactos.\ninput_sha256.csv: fuentes y huellas actuales. visual_review.csv y visual_checks/:\nevidencia de revisión parcial. maul_training_clips.csv: las 14 filas y el corte.\nLa tabla maestra posterior y el manifiesto de regeneración se usan solo para\nverificar procedencia. No se usan sus particiones, modelos ni clips regenerados\npara evaluar. El modelo aquí analizado precede a M0–M3 de partidos completos.\n\nReproducir desde la raíz del proyecto:\n  .venv/Scripts/python.exe models/common/studies/initial_clip_error_analysis_20261017/analyze.py\nEl análisis usa solamente la biblioteca estándar. No necesita TensorFlow.\nLas planchas conservadas se extrajeron con ffmpeg, fps=1, scale=400:-1, tile=3x2,\nprimer mosaico, a partir de data/7seg unique; las *_model_input.png usan los\nvideos de data/videos/clips_224, scale=224:224. No se editaron videos.\n'
    report = report.replace('0.787026767', f'{macro_f1:.9f}').replace('0.816868 aproximadamente', f'{weighted_f1:.9f}')
    (OUT / 'report.txt').write_text(report, encoding='utf-8')
    tex = ['% Generated analysis only. Not included in the active thesis.', '\\begin{table}[htbp]', '\\centering', '\\small', '\\caption{Errores del CNN-RNN de referencia según clase y rótulo de origen de los clips. Los subtipos de otro proceden de nombres originales, no de una anotación visual exhaustiva.}', '\\begin{tabular}{lrrrrr}', '\\hline', 'Clase u origen & Clips & Otro & Scrum & Lineout & Error (\\%) \\\\', '\\hline']
    for r in summary:
        tex.append(f"{r['category_display']} & {r['n']} & {r['pred_other']} & {r['pred_scrum']} & {r['pred_line']} & {r['error_rate_pct']:.2f} \\\\")
    tex.extend(['\\hline', 'Total & 817 & 467 & 105 & 245 & 18.85 \\\\', '\\hline', '\\end{tabular}', '\\end{table}'])
    (OUT / 'table_for_thesis.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print(json.dumps(dict(samples=817, errors=154, accuracy=accuracy, f1_macro=macro_f1, output=str(OUT)), ensure_ascii=False))
    print(table(summary))
if __name__ == '__main__':
    main()
