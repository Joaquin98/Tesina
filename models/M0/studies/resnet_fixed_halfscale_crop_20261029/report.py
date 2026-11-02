"""Report the wider fixed-halfscale field-of-view experiment only."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import csv, json
from pathlib import Path
HERE = experiment_directory(__file__)
OUT = HERE / 'outputs'

def load(path):
    return json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8-sig')) if _resolve_model_artifact(path).exists() else None

def report():
    reuse = load(OUT / 'reference224_reuse.json')
    audit = load(OUT / 'source_audit.json')
    protocol = load(OUT / 'protocol_snapshot.json')
    complete = load(OUT / 'execution_complete.json')
    stop = load(OUT / 'user_stop.json')
    failure = load(OUT / 'last_failure.json')
    counts = {s: len(list(_model_artifact_glob(OUT / 'runs', f'M0_R{s}_*/complete.json', recursive=False))) for s in (224, 320)}
    lines = ['ComparaciÃ³n de campo visual con escalado fijo x0.5', 'ComparaciÃ³n iniciada el 29 de octubre de 2026.', '']
    if stop:
        lines += ['Estado: ejecuciÃ³n detenida.']
    elif complete and (load(OUT / 'final_validation.json') or {}).get('status') == 'passed':
        lines += ['Estado: comparaciÃ³n completa y validaciÃ³n final aprobada (36 evaluaciones).']
    elif complete:
        lines += ['Estado: cÃ¡lculo completo. Revisar validaciÃ³n final de artefactos.']
    elif failure:
        lines += ['Estado: se registrÃ³ un fallo. Consultar outputs/last_failure.json y el estado del proceso antes de interpretar la ejecuciÃ³n.']
    else:
        lines += ['Estado: preparaciÃ³n o ejecuciÃ³n en curso. TodavÃ­a no hay una comparaciÃ³n completa de calidad.']
    lines += [f'Evaluaciones completas:224={counts[224]}/18;320={counts[320]}/18.', '', 'QuÃ© se compara', 'Desde el mismo fotograma original se aplica realmente cv2.resize con fx=fy=0.5 e INTER_LINEAR. Luego se recorta el centro a224x224 o320x320. No hay un segundo redimensionado.', 'Para una fuente1280x720, el fotograma intermedio mide640x360.224 cubre448x448 pÃ­xeles de la fuente y320 cubre640x640. La escala de los objetos es idÃ©ntica y el Ã¡rea cubierta por320 es2.0408 veces la de224.', 'La condiciÃ³n de320 incorpora mÃ¡s contexto. No es la comparaciÃ³n anterior de campo visual fijo redimensionado.', 'La referencia224 reutiliza copias de la cachÃ©, tres iniciales y18 evaluaciones anteriores tras demostrar equivalencia de fuentes, pÃ­xeles, caracterÃ­sticas, mÃ¡scaras, particiones, arquitectura y entrenamiento.320 se extrae y entrena desde su representaciÃ³n propia. La prueba estÃ¡ documentada en reference224_reuse.json.' if reuse else 'Ambas condiciones regeneran caracterÃ­sticas, entrenamientos iniciales y adaptaciones. No se reutilizan modelos, cachÃ©s o mÃ©tricas del experimento anterior.', 'La carpeta del experimento anterior, sus resultados remotos y el448 detenido permanecen intactos. No se modifica la tesina, presentaciÃ³n o producciÃ³n.', '', 'Figura de encuadre', 'outputs/geometry/field_of_view_comparison.png muestra tres fuentes histÃ³ricas y los seis grupos de partidos. Verde:224. Amarillo:320. Los recortes se muestran a escala1:1 y el centro224 del recorte320 es idÃ©ntico pÃ­xel a pÃ­xel al recorte224.']
    early = load(OUT / 'early_check.json')
    if early:
        lines += ['', 'Primera etapa definitiva (semilla17)', 'ValidaciÃ³n inicial de360 clips, utilizada tambiÃ©n por EarlyStopping. No es F1 temporal de partidos ni una evaluaciÃ³n independiente.', 'Recorte | accuracy | F1 macro (3 clases) | F1 line | F1 scrum']
        for size, r in early['conditions'].items():
            b = r['metrics']
            lines.append(f"{size} | {b['accuracy']:.4f} | {b['f1_macro']:.4f} | {b['per_class']['line']['f1']:.4f} | {b['per_class']['scrum']['f1']:.4f}")
        lines += ['Las caracterÃ­sticas y el checkpoint320 semilla17 se reutilizan sin repetir el entrenamiento en el total. No se adapta el protocolo a este resultado.']
    else:
        lines += ['', 'Primera etapa: se priorizan los clips de la particiÃ³n inicial fija y el entrenamiento definitivo320 semilla17. Se incorporan despuÃ©s a la cachÃ© y a las evaluaciones completas.']
    if audit:
        lines += ['', 'AuditorÃ­a', f"Fuentes originales:{audit['sources']}. Ejemplos:{audit['examples']} ({audit['historical']} histÃ³ricos y{audit['final']} de partidos). Fotogramas por condiciÃ³n:{audit['source_samples']}.", f"Estado:{audit['status']}. Casos de fuente incompatibles:{len(audit['source_failures'])}. Discrepancias temporales:{len(audit['temporal_failures'])}.", 'No se excluyen fuentes ni se rellena o amplÃ­a una imagen para hacer caber320. Una incompatibilidad bloquea el cÃ¡lculo.']
    lines += ['', 'Modelo y protocolo', 'M0-H2-D2. ResNet-50 con ImageNet congelada y pooling global2048. BiGRU64 por direcciÃ³n, BiGRU32 por direcciÃ³n, Dropout0.3, Dense64 ReLU y Dense3 Softmax.', 'Semillas emparejadas17,42,2026. Los mismos ejemplos y orden de particiÃ³n inicial, instantes y mÃ¡scaras para ambos recortes. Padding temporal hasta200, muestreo25FPS.', 'Entrenamiento inicial propio por recorte y semilla.1440 filas de entrenamiento y360 de validaciÃ³n, misma inicializaciÃ³n por semilla, Adam1e-4, batch16, mÃ¡ximo100 Ã©pocas, EarlyStopping val_accuracy paciencia10 y ReduceLR val_loss paciencia5.', 'AdaptaciÃ³n independiente desde ese inicial: CH completo mÃ¡s cuatro partidos, validaciÃ³n en uno y prueba en otro. Adam1e-4, batch16, mÃ¡ximo40 Ã©pocas y EarlyStopping val_loss paciencia7. Se adaptan todos los pesos posteriores a ResNet.', 'Umbrales elegidos sÃ³lo en validaciÃ³n con la rejilla original. No se ajustan hiperparÃ¡metros con prueba. M1 no forma parte de este experimento.']
    if protocol:
        lines += ['Pliegue | entrenamiento CV | validaciÃ³n | prueba']
        for f in protocol['folds']:
            lines.append(f"{f['fold']} | {','.join(f['train_groups'])} | {f['development_group']} | {f['test_group']}")
    lines += ['', 'MÃ©tricas e interpretaciÃ³n', 'Se utiliza models/common/comparison/metrics.py. Los eventos agrupan bloques consecutivos de la misma clase por secciÃ³n y partido. El ancla real es el centro del evento; el ancla predicha corresponde al bloque de mayor probabilidad.', 'Emparejamiento uno a uno con crÃ©dito max(0,1-distancia/35) hasta35 segundos. Precision temporal divide crÃ©dito por eventos predichos, recall por eventos reales y F1 por la suma de ambos denominadores.', 'FP lejanos por hora usa la definiciÃ³n original: predicciones sin evento real de su clase a35 segundos. Se guardan tambiÃ©n todos los eventos no emparejados por hora, incluidos duplicados cercanos.', 'Las semillas se emparejan dentro del mismo partido. Se promedian por partido y se da igual peso a los seis partidos. El bootstrap remuestrea seis partidos completos, no18 casos independientes. Sus intervalos son descriptivos y condicionados al corpus.']
    if complete and _resolve_model_artifact(OUT / 'per_match_seed.csv').exists():
        rows = list(csv.DictReader(_resolve_model_artifact(OUT / 'per_match_seed.csv').open(encoding='utf-8')))
        lines += ['', 'Resultados medidos', 'Recorte | n | F1 macro temporal | F1 line | P line | R line | F1 scrum | P scrum | R scrum | FP lejanos/h (suma)']
        means = {}
        for s in (224, 320):
            r = [v for v in rows if int(v['size']) == s]
            assert len(r) == 18
            cols = ['f1_temporal_macro', 'line_weighted_f1', 'line_weighted_precision', 'line_weighted_recall', 'scrum_weighted_f1', 'scrum_weighted_precision', 'scrum_weighted_recall', 'line_far_false_positives_per_hour', 'scrum_far_false_positives_per_hour']
            mean = {k: sum((float(v[k]) for v in r)) / len(r) for k in cols}
            means[s] = mean
            lines.append(f'{s} |18| ' + ' | '.join((f'{mean[k]:.4f}' for k in cols[:7])) + f' | {mean[cols[-2]] + mean[cols[-1]]:.3f}')
        paired = load(OUT / 'paired_summary.json')['comparisons'][0]
        lo, hi = paired['cluster_bootstrap_ci95']
        lines += [f"Delta320-224 F1 temporal macro:{paired['mean_delta']:.5f}. Intervalo descriptivo95% por partido:[{lo:.5f},{hi:.5f}]. Partidos positivos:{paired['positive_matches']}/6. Semillas positivas:{paired['positive_seeds']}/3.", f"Delta F1 line:{paired['line_mean_delta']:.5f}. Delta F1 scrum:{paired['scrum_mean_delta']:.5f}."]
        lines += ['SeÃ±al consistente segÃºn el criterio fijado: ' + ('sÃ­' if paired['promising'] else 'no') + '.']
        if paired['promising']:
            lines += ['ConclusiÃ³n de calidad:320 muestra una mejora consistente bajo este protocolo, condicionada a los seis partidos. Valorar el costo adicional de extracciÃ³n antes de adoptarlo.']
        else:
            lines += ['ConclusiÃ³n: esta comparaciÃ³n no aporta una mejora consistente que justifique reemplazar224 por320 bajo este protocolo. No se atribuye un Ã³ptimo universal a224.']
        lines += ['', 'Costo medido por condiciÃ³n', 'Recorte | extracciÃ³n activa(h) | CNN(h) | decode+halfscale+crop(h) | inicial(h) | adaptaciÃ³n/evaluaciÃ³n(h) | procesamiento total(h) | RSS mÃ¡x(GiB) | cachÃ©(GiB)']
        ledger = [json.loads(line) for line in _resolve_model_artifact(OUT / 'worker_timings.jsonl').read_text().splitlines()] if _resolve_model_artifact(OUT / 'worker_timings.jsonl').exists() else []
        for s in (224, 320):
            cache = load(OUT / f'cache/R{s}/state.json')
            initial = [load(p) for p in _model_artifact_glob(OUT / 'initial', f'R{s}_S*/complete.json', recursive=False)]
            runs = [load(p) for p in _model_artifact_glob(OUT / 'runs', f'M0_R{s}_*/complete.json', recursive=False)]
            stage = load(OUT / f'initial_stage/R{s}/state.json') or {}
            cost = {key: cache[key] + stage.get(key, 0) for key in ('active_s', 'inference_s', 'decode_resize_s')}
            ih = sum((r['elapsed_s'] for r in initial)) / 3600
            rh = sum((r['elapsed_s'] for r in runs)) / 3600
            eh = cost['active_s'] / 3600
            wall = sum((t['wall_s'] for t in ledger if '--size' in t['command'] and int(t['command'][t['command'].index('--size') + 1]) == s)) / 3600
            peak = max([cache['peak_rss_bytes'], stage.get('peak_rss_bytes', 0)] + [r['peak_rss_bytes'] for r in initial + runs]) / 2 ** 30
            lines.append(f"{s} | {eh:.3f} | {cost['inference_s'] / 3600:.3f} | {cost['decode_resize_s'] / 3600:.3f} | {ih:.3f} | {rh:.3f} | {wall:.3f} | {peak:.3f} | {cache['features_bytes'] / 2 ** 30:.3f}")
        lines += ['En320, extracciÃ³n suma la etapa inicial y el resto de la cachÃ©. Las filas copiadas de la primera etapa no se vuelven a procesar por ResNet. El procesamiento total incluye los workers completos, incluida la comprobaciÃ³n inicial. La cachÃ© auxiliar de la primera etapa tambiÃ©n ocupa espacio y se conserva para auditar la reutilizaciÃ³n.']
        if reuse:
            lines += ['Para224, los tiempos de extracciÃ³n, entrenamiento y adaptaciÃ³n y el RSS proceden de la referencia histÃ³rica reutilizada; no son mediciones nuevas del recorrido completo x0.5. La columna procesamiento total cuenta el trabajo de esta nueva ejecuciÃ³n: repeticiÃ³n parcial interrumpida, copia, integridad e importaciÃ³n. No sumar ese total a los tiempos histÃ³ricos. Los campos training_s, prediction_s y threshold_and_metrics_s no medidos originalmente se conservan como nulos.320 tiene todos los tiempos nuevos medidos.']
        lines += ['El total proviene del reloj de cada proceso del registro worker_timings.jsonl e incluye importaciÃ³n, guardas, extracciÃ³n, entrenamiento, predicciÃ³n, umbrales y escritura. La auditorÃ­a comÃºn se informa aparte. Los apagados e intentos interrumpidos antes de registrar su salida pueden no estar completamente medidos. Los tiempos de entrenamiento, predicciÃ³n y bÃºsqueda de umbrales se guardan por corrida en complete.json. RSS incluye pÃ¡ginas de cachÃ© mapeadas y no mide exclusivamente el modelo.']
    else:
        lines += ['', 'Calidad temporal en partidos pendiente', 'La primera etapa sÃ³lo informa validaciÃ³n inicial de clips. Hasta completar las evaluaciones de partidos no permite concluir si el campo visual mayor mejora el F1 temporal.']
    lines += ['', 'Costo preliminar y entorno']
    for s in (224, 320):
        pilot = load(OUT / f'pilot/R{s}.json')
        state = load(OUT / f'cache/R{s}/state.json')
        if pilot:
            lines.append(f"Recorte{s}: piloto compilado de16 fotogramas, {pilot['s_per_frame']:.6f}s/fotograma y RSS {pilot['peak_rss_bytes'] / 2 ** 30:.3f}GiB. ExtrapolaciÃ³n CNN sin decode ni entrenamiento: {pilot['s_per_frame'] * (audit['source_samples'] if audit else 0) / 3600:.1f}h.")
            if s == 224:
                lines.append('Runtime/hardware: ' + json.dumps(pilot['runtime'], ensure_ascii=False) + '; ' + json.dumps(pilot['hardware'], ensure_ascii=False) + f"; hilos TensorFlow:{pilot['tensorflow_threads']}; interop1.")
        staged = load(OUT / f'initial_stage/R{s}/state.json')
        if staged:
            lines.append(f"Primera etapa{s}: {staged['done']}/1561 ejemplos Ãºnicos, {staged['frames']} fotogramas y {staged['active_s'] / 3600:.2f}h activas.")
        if state:
            origin = 'Referencia reutilizada; horas histÃ³ricas de extracciÃ³n' if state.get('artifact_origin') == 'reused_equivalent_previous224' else 'ExtracciÃ³n nueva; horas activas'
            lines.append(f"CachÃ©{s}: {state['done']}/8796 ejemplos, {state['frames']} fotogramas. {origin}: {state['active_s'] / 3600:.2f}h.")
    lines += ['', 'Limitaciones', 'La particiÃ³n inicial preserva239 nombres duplicados por balanceo y65 nombres compartidos entre entrenamiento y validaciÃ³n. DuraciÃ³n histÃ³rica mediana6.08s, no se fuerza7s.', 'D2 conserva todo CH y puede compartir fuentes con CV (F17,F3,F10). La separaciÃ³n interna de partidos CV se conserva, pero no se ha probado independencia absoluta entre CH y CV.', 'SÃ³lo hay seis partidos y los entrenamientos de pliegues se superponen. No se afirma generalizaciÃ³n a partidos externos.', 'Un encuadre mÃ¡s amplio puede incorporar marcador o texto y otras pistas de transmisiÃ³n. Las figuras se conservan para examinar ese efecto. Se sigue exactamente el recorte central definido.', '', 'ReproducciÃ³n y trazabilidad', 'python preflight.py audit; python preflight.py visual; python pilot.py 224; python pilot.py 320; python validate_contract.py; python experiment.py prepare; python experiment.py all; python report.py. Usar los argumentos separados por espacios.', 'config.json y outputs/protocol_snapshot.json fijan parÃ¡metros y hashes. outputs/source_audit.csv y temporal_audit.csv documentan fuentes e instantes. outputs/runs contiene modelos, probabilidades, umbrales, mÃ©tricas y hashes por partido/semilla.', 'SÃ³lo la referencia224 equivalente y documentada se reutiliza en esta comparaciÃ³n. No mezclar la antigua variante320 a campo fijo ni la simulaciÃ³n o el resultado remoto informado de448 con esta prueba.']
    if complete and (load(OUT / 'final_validation.json') or {}).get('status') == 'passed':
        from models.M0.studies.resnet_fixed_halfscale_crop_20261029.report_details import extend_report
        extend_report(lines)
    path = HERE / 'informe.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(str(path))
if __name__ == '__main__':
    report()
