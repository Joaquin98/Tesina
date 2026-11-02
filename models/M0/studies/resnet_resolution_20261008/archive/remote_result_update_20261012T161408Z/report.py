"""Render honest engineering or completed scientific report from local artifacts."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import csv, json
HERE = experiment_directory(__file__)
OUT = HERE / 'outputs'

def load(path):
    return json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8-sig')) if _resolve_model_artifact(path).exists() else None

def report():
    if load(OUT / 'user_stop.json') and _resolve_model_artifact(HERE / 'cierre_y_escenario_hipotetico.txt').exists():
        closed = _resolve_model_artifact(HERE / 'cierre_y_escenario_hipotetico.txt').read_text(encoding='utf-8')
        closed += '\nEl protocolo detallado anterior se conserva en archive/protocol_report_before_user_stop.txt.\n'
        (HERE / 'informe.txt').write_text(closed, encoding='utf-8')
        print(str(HERE / 'informe.txt'))
        return
    smoke = load(OUT / 'smoke/smoke.json')
    audit = load(OUT / 'source_audit.json')
    protocol = load(OUT / 'protocol_snapshot.json')
    complete = load(OUT / 'execution_complete.json')
    exit_state = load(OUT / 'job_monitor_exit.json')
    lines = ['ComparaciÃ³n controlada de resoluciones de ResNet-50', 'Experimento propio iniciado el 8 de octubre de 2026.', '']
    if complete:
        lines += ['Estado: ejecuciÃ³n cientÃ­fica completa.']
    elif audit and audit['status'] == 'blocked':
        lines += ['Estado: bloqueado por discrepancias verificadas de los recursos fuente.']
    elif exit_state and exit_state['status'] != 'complete':
        lines += ['Estado: ejecuciÃ³n interrumpida o fallida. Ver last_failure.json y los logs para el motivo concreto.']
    else:
        lines += ['Estado: preparaciÃ³n o ejecuciÃ³n en curso. No existe todavÃ­a una comparaciÃ³n cientÃ­fica completa.']
    lines += ['Objetivo: comparar 224, 320 y 448 entre las condiciones evaluadas y medir su costo. No determinar un Ã³ptimo universal.', 'Los resultados se guardan en el directorio de este experimento.', '', 'Protocolo y diferencias reconciliadas', 'Se usa la misma regiÃ³n nativa anterior a la reducciÃ³n a224, sin ampliar imÃ¡genes previamente reducidas. INTER_LINEAR a las tres resoluciones, BGR aRGB, preprocess_input de ResNet-50, ImageNet congelado y pooling global de2048 valores.', 'HistÃ³rico: se invierte la transformaciÃ³n real scale0.5 +crop central224. Partidos: se invierte reference_height360 +crop central224. Cada dominio mantiene su propio encuadre previo.', 'Se reconstruye tambiÃ©n224 desde originales. La recompresiÃ³n de los viejos224 puede cambiar pÃ­xeles y caracterÃ­sticas, por lo que sus resultados anteriores no se sustituyen ni se mezclan como referencia emparejada.', 'Muestreo histÃ³rico: reproduce los tiempos float32 de load_video a25 FPS y mÃ¡ximo200. La duraciÃ³n real mediana es6.08 segundos, el mÃ­nimo0.04 y el mÃ¡ximo7.0. Solo408 de4082 clips tienen175 muestras vÃ¡lidas. Se conservan exactamente todas las mÃ¡scaras, incluidas las secuencias cortas. Partidos: reproduce sampled_block_indices, incluido recorte al Ãºltimo fotograma de las secciones.', 'Cabeza: BiGRU64 por direcciÃ³n con secuencia, BiGRU32 por direcciÃ³n con resumen, Dropout0.3, Dense64 ReLU, Dense3 Softmax.847235 parÃ¡metros.', 'Semillas emparejadas:17,42,2026. Misma inicializaciÃ³n verificable por hash entre resoluciones para cada semilla.', 'Entrenamiento inicial: balanced_train en el mismo orden, Ãºltimo20% para validaciÃ³n, Adam1e-4 sin weight decay, batch16, mÃ¡ximo100 Ã©pocas, parada por val_accuracy con paciencia10, reducciÃ³n de tasa por val_loss con paciencia5 y mÃ­nimo1e-6. train_best dice50 en el encabezado pero ejecuta100 por defecto. La inicializaciÃ³n histÃ³rica no impone semilla, aquÃ­ se fija por condiciÃ³n emparejada.', 'La validaciÃ³n inicial conserva el mÃ©todo histÃ³rico, que no agrupa por fuente y comparte65 nombres con entrenamiento tras el balance. D2 conserva CH completo en todos los pliegues. Los cÃ³digos histÃ³ricos F17,F3,F10 coinciden con nombres fuente de V_1,V_3,V_6, por lo que no se acreditÃ³ independencia de fuentes entre CH y CV. Esta limitaciÃ³n se declara. Dentro de CV, los seis pliegues mantienen partidos disjuntos. La comparaciÃ³n reproduce esa definiciÃ³n ejecutable de D2 y no prueba generalizaciÃ³n a partidos completamente nuevos.', 'AdaptaciÃ³n M0-H2-D2: todos los pesos posteriores aResNet, CH completo mÃ¡s cuatro partidos. Adam1e-4, batch16, mÃ¡ximo40 Ã©pocas, parada por val_loss con paciencia7. Se verificaron estos valores contra run.json de la referencia.', 'Se utilizan los mismos seis pliegues PV01 aPV06, cada uno con cuatro partidos de entrenamiento, uno de validaciÃ³n y uno de prueba.', 'Las cachÃ©s se extraen una vez por resoluciÃ³n y se reutilizan para todas las semillas y pliegues. Los procesos de extracciÃ³n y entrenamiento quedan aislados.', '', 'MÃ©tricas reales de la evaluaciÃ³n', 'Los eventos agrupan bloques consecutivos de una misma clase dentro de cada secciÃ³n y partido. El ancla real es el centro del evento. El ancla predicha es el centro del bloque con mayor probabilidad, con desempate por comienzo.', 'El emparejamiento uno auno maximiza el crÃ©dito temporal. Hasta35s de distancia, el crÃ©dito es max(0,1-distancia/35). La suma de crÃ©ditos se usa como TP ponderado. Precision temporal divide por eventos predichos, recall por eventos reales y F1 por la suma de ambos denominadores.', 'Se informan falsas alarmas lejanas por hora, que excluyen duplicados a<=35s de algÃºn evento real, segÃºn metrics.py. TambiÃ©n se guardan todos los eventos predichos no emparejados por hora para evitar confundir ambas definiciones.', 'La exposiciÃ³n temporal es la suma del fin de cada secciÃ³n, igual que la evaluaciÃ³n existente.', 'Umbrales: rejilla0.10 a0.95, paso0.05, elegida exclusivamente en validaciÃ³n. Primero maximiza F1 temporal macro de line y scrum, luego el menor F1 por bloque de esas clases y despuÃ©s minimiza falsas alarmas lejanas. Se reutiliza la implementaciÃ³n original.', 'La predicciÃ³n usa lotes de64 y normalize_probabilities antes de registrar probabilidades y anclar eventos, igual que la implementaciÃ³n original. La correcciÃ³n de compatibilidad se aplicÃ³ antes de cualquier entrenamiento o evaluaciÃ³n de prueba. La instantÃ¡nea previa se conserva en archive/. Los resultados de prueba no modifican el entrenamiento, el muestreo ni los umbrales.', '', 'Prueba pequeÃ±a de extracciÃ³n']
    if smoke:
        lines += ['ResoluciÃ³n | cuadros | segundos/cuadro | segundos de inferencia | RSS mÃ¡ximo del proceso (GiB)']
        for r in smoke['results']:
            lines.append(f"{r['resolution']} | {r['n_frames']} | {r['s_per_frame']:.6f} | {r['inference_s']:.3f} | {r['peak_rss_bytes'] / 2 ** 30:.3f}")
        lines += ['Estas mediciones usan16 cuadros:8 de un clip histÃ³rico y8 de un bloque de partido. Sirven para verificar funcionamiento, no calidad.', 'El benchmark pequeÃ±o fue secuencial en un mismo proceso. TensorFlow puede retener memoria de condiciones anteriores. Los RSS de esta tabla no estiman incrementos puros por resoluciÃ³n. Las mediciones completas se hacen en procesos independientes.', 'La inferencia pequeÃ±a fue eager. La extracciÃ³n completa usa funciÃ³n compilada, por lo que la extrapolaciÃ³n es preliminar.', 'TensorFlow' + smoke['tensorflow'] + ', Keras' + smoke['keras'] + ', OpenCV' + smoke['opencv'] + '. CPU Ãºnicamente,16 procesadores lÃ³gicos y aproximadamente31.39GiB RAM.']
    pilots = [load(OUT / 'compiled_pilot' / f'R{s}.json') for s in (224, 320, 448)]
    if all(pilots):
        lines += ['', 'Prueba compilada con procesos independientes', 'ResoluciÃ³n | segundos/cuadro | RSS mÃ¡ximo del proceso (GiB) | diferencia mÃ¡xima de caracterÃ­sticas frente a eager']
        for r in pilots:
            lines.append(f"{r['resolution']} | {r['s_per_frame']:.6f} | {r['peak_rss_bytes'] / 2 ** 30:.3f} | {r['max_abs_vs_eager']:.8f}")
    if audit:
        lines += ['', 'AuditorÃ­a de fuentes', f"HistÃ³ricos: {audit['historical']}. Bloques de partidos: {audit['final']}. Casos con fallos: {audit['failures']}.", f"Tiempo de auditorÃ­a: {audit['source_audit_seconds']:.1f}s. Instantes totales aextraer por resoluciÃ³n: {audit['source_samples']}."]
        if audit['failures']:
            lines += ['Detalle exacto: source_audit.csv y source_audit.json.', json.dumps(audit['failure_counts'], ensure_ascii=False)]
    if protocol:
        lines += ['', 'Pliegues congelados', 'Pliegue | entrenamiento | validaciÃ³n | prueba']
        for f in protocol['folds']:
            lines.append(f"{f['fold']} | {','.join(f['train_groups'])} | {f['development_group']} | {f['test_group']}")
        lines += ['', 'Costo y reproducciÃ³n', 'ProyecciÃ³n preliminar de extracciÃ³n segÃºn la prueba pequeÃ±a, sin entrenamiento ni decodificaciÃ³n:']
        for size, hours in protocol['smoke_extraction_projection_hours'].items():
            lines.append(f'{size}: {hours:.1f} horas. Es una extrapolaciÃ³n, no tiempo total medido.')
    rows = []
    if _resolve_model_artifact(OUT / 'per_match_seed.csv').exists():
        with _resolve_model_artifact(OUT / 'per_match_seed.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
    lines += ['', 'Resultados de calidad']
    if not complete:
        lines += ['No hay resultados completos que permitan recomendar una resoluciÃ³n por calidad. Las salidas parciales, si existen, no demuestran ausencia o presencia de mejora.']
    else:
        import pandas as pd
        frame = pd.DataFrame(rows)
        numeric = [v for v in frame.columns if v not in ('model', 'fold', 'test_group')]
        for col in numeric:
            frame[col] = pd.to_numeric(frame[col])
        lines += ['Modelo | resoluciÃ³n | corridas | F1 temporal macro | F1 line | P line | R line | F1 scrum | P scrum | R scrum | FP lejanos/h (suma clases)']
        for (model, size), part in frame.groupby(['model', 'size']):
            means = part.mean(numeric_only=True)
            v = [means[k] for k in ['f1_temporal_macro', 'line_weighted_f1', 'line_weighted_precision', 'line_weighted_recall', 'scrum_weighted_f1', 'scrum_weighted_precision', 'scrum_weighted_recall']]
            far = means.line_far_false_positives_per_hour + means.scrum_far_false_positives_per_hour
            lines.append(f'{model} | {size} | {len(part)} | ' + ' | '.join((f'{x:.4f}' for x in v)) + f' | {far:.3f}')
        pairs = load(OUT / 'paired_summary.json')
        lines += ['', 'Diferencias emparejadas con224', 'Modelo | resoluciÃ³n | delta F1 macro | IC descriptivo95% por partido | partidos con delta positivo | semillas con delta positivo']
        for r in pairs['comparisons']:
            lo, hi = r['cluster_bootstrap_ci95']
            lines.append(f"{r['model']} | {r['size']} | {r['mean_delta']:.4f} | [{lo:.4f},{hi:.4f}] | {r['positive_matches']}/6 | {r['positive_seeds']}/3")
        candidates = [r for r in pairs['comparisons'] if r['model'] == 'M0' and r['promising']]
        confirmations = [r for r in pairs['comparisons'] if r['model'] == 'M1' and r['promising']]
        if not candidates:
            lines += ['RecomendaciÃ³n: conservar224. Ninguna resoluciÃ³n mayor cumpliÃ³ el criterio previo de mejora consistente en M0 y el costo de extracciÃ³n de224 es menor. Esto no demuestra un Ã³ptimo universal.']
        elif confirmations:
            chosen = max(confirmations, key=lambda r: r['mean_delta'])
            lines += [f"RecomendaciÃ³n condicional: {chosen['size']} frente a224, con delta medio de F1 temporal macro enM1 de {chosen['mean_delta']:.4f}. La seÃ±al cumple el criterio predefinido tambiÃ©n enM1, pero debe sopesarse contra el costo completo medido. Los mismos seis partidos y las limitaciones deD2 restringen esta conclusiÃ³n."]
        else:
            lines += ['M0 mostrÃ³ una seÃ±al prometedora que no se confirmÃ³ consistentemente en M1. Conservar224 para el procedimiento combinado, con la evidencia condicionada a estos partidos.']
        lines += ['', 'Costos completos medidos', 'ResoluciÃ³n | extracciÃ³n total activa(h) | inferencia CNN(h) | decodificaciÃ³n y resize(h) | RSS extracciÃ³n(GiB) | cachÃ©(GiB)']
        for size in (224, 320, 448):
            state = load(OUT / f'cache/R{size}/state.json')
            lines.append(f"{size} | {state['active_s'] / 3600:.3f} | {state['inference_s'] / 3600:.3f} | {state['decode_resize_s'] / 3600:.3f} | {state['peak_rss_bytes'] / 2 ** 30:.3f} | {state['features_bytes'] / 2 ** 30:.3f}")
        lines += ['Tiempo de adaptaciÃ³n y memoria por partido y semilla: per_match_seed.csv. Tiempo de entrenamiento inicial: initial/Rxxx_Sxxx/complete.json. Los costos de la rama geomÃ©trica compartida deM1 se deben asignar por igual a ambas resoluciones al comparar el costo marginal.']
    lines += ['', 'Incertidumbre y confirmaciÃ³n', 'Se promedian primero las semillas de cada partido y se da igual peso a los seis partidos. Las diferencias usan pares de la misma semilla y pliegue. El bootstrap re-muestrea partidos completos, conservando el emparejamiento.', 'Las tres semillas no crean18 partidos independientes. Los conjuntos de entrenamiento de los pliegues se superponen. Con solo seis partidos, los intervalos son descriptivos y condicionados al corpus.', 'El criterio de seÃ±al prometedora se fijÃ³ antes de mirar las pruebas: delta temporal macro medio positivo, positivo en las tres semillas y al menos cuatro de seis partidos, con delta medio de cada clase no menor que-0.01.', 'Si M0 cumple el criterio, la mayor mejora se compara contra224 enM1-H2-D2. La rama YOLO queda fija y se reutilizan sus mismas caracterÃ­sticas y el mismo bosque geomÃ©trico por semilla y pliegue. La elecciÃ³n del candidato utiliza las pruebas M0 existentes y por eso es exploratoria.', 'Solo V_1 a V_6 tienen cobertura completa de anotaciones. validation_unseen contiene un fragmento de un minuto cuya identidad independiente no se acreditÃ³. No hay allÃ­ un partido completo con cobertura de anotaciones que permita una confirmaciÃ³n externa. Una confirmaciÃ³n externa necesitarÃ­a un partido no utilizado en las decisiones. No se atribuyen resultados inventados a videos no evaluados.', '', 'CÃ³mo reproducir', 'Usar Python313 con TensorFlow2.20 y Keras3.13.2, desde D:/Tesina.', 'python models/M0/studies/resnet_resolution_20261008/preflight.py smoke', 'python models/M0/studies/resnet_resolution_20261008/preflight.py audit', 'python models/M0/studies/resnet_resolution_20261008/validate_contract.py', 'python models/M0/studies/resnet_resolution_20261008/experiment.py prepare', 'python models/M0/studies/resnet_resolution_20261008/experiment.py all', 'python models/M0/studies/resnet_resolution_20261008/report.py', 'La ejecuciÃ³n completa reanuda cachÃ©s y omite corridas completas. Si falla, last_failure.json identifica el comando y los logs conservan el error. No se deben cambiar cÃ³digo, configuraciÃ³n o manifiestos despuÃ©s deprepare para continuar la misma comparaciÃ³n.', '', 'LimitaciÃ³n del entorno de ejecuciÃ³n', 'El entorno de ejecuciÃ³n aislado fallÃ³ antes de iniciar el programa. La ejecuciÃ³n local permitiÃ³ continuar. Este error no indica falta de videos o TensorFlow.']
    (HERE / 'informe.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(str(HERE / 'informe.txt'))
if __name__ == '__main__':
    report()
