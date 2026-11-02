"""Prepare frozen-policy comparisons and candidate evidence; never fabricate review labels."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from typing import Dict, List, Optional
import ast, json, csv, html
import numpy as np
P = experiment_directory(__file__)

def main():
    cfg = json.loads(_resolve_model_artifact(P / 'inputs/config.json').read_text())
    props = json.loads(_resolve_model_artifact(P / 'inputs/video_properties.json').read_text())
    windows = [json.loads(x) for x in _resolve_model_artifact(P / 'results/windows.jsonl').read_text().splitlines()]
    original = json.loads(_resolve_model_artifact(P / 'results/events.json').read_text())
    source = _resolve_model_artifact(P.parents[1] / 'models/M0/web_adapter.py')
    tree = ast.parse(_resolve_model_artifact(source).read_text(encoding='utf-8-sig'))
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and any((isinstance(m, ast.FunctionDef) and m.name == '_post_process_event_detections' for m in n.body))))
    methods = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name in {'_post_process_event_detections', '_event_has_support', '_finalize_event_detection'}]
    mod = ast.Module(body=[ast.ClassDef(name='Policy', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ns = dict(np=np, Dict=Dict, List=List, Optional=Optional)
    exec(compile(ast.fix_missing_locations(mod), '<frozen postprocessing>', 'exec'), ns)
    policy = ns['Policy']()
    for k, v in cfg['event_policy'].items():
        setattr(policy, k, v)
    baseline = policy._post_process_event_detections(windows, cfg['thresholds']['scrum'], props['duration_s'], props['fps'], {'line': cfg['thresholds']['line']})
    assert baseline == original
    policy.EVENT_REQUIRE_TARGET_WINNER = True
    argmax = policy._post_process_event_detections(windows, 0.0, props['duration_s'], props['fps'], {'line': 0.0})
    out = P / 'argmax_comparison'
    out.mkdir(exist_ok=True)
    (out / 'events_argmax.json').write_text(json.dumps(argmax, indent=2), encoding='utf-8')

    def counts(events):
        return {k: sum((e['play_type'] == k for e in events)) for k in ['line', 'scrum']}
    result = dict(original=counts(original), argmax=counts(argmax), baseline_exactly_reproduced=True, manual_precision='pending visual review', recall='pending independent full-match inventory', video_duration_s=props['duration_s'])
    kickoff = json.loads(_resolve_model_artifact(P / 'inputs/kickoff.json').read_text(encoding='utf-8'))['source_time_s']
    inventory = json.loads(_resolve_model_artifact(P / 'full_match_review/initial_inventory.json').read_text(encoding='utf-8'))
    matches = []
    for truth in inventory['events']:
        ids = [i + 1 for i, e in enumerate(argmax) if e['play_type'] == truth['play_type'] and min(e['end_time'], truth['end_s']) > max(e['start_time'], truth['start_s'])]
        matches.append(dict(inventory_id=truth['id'], play_type=truth['play_type'], start_s=truth['start_s'], end_s=truth['end_s'], argmax_candidate_ids=ids))
    result['after_kickoff'] = dict(kickoff_s=kickoff, original=counts([e for e in original if e['end_time'] > kickoff]), argmax=counts([e for e in argmax if e['end_time'] > kickoff]))
    result['provisional_inventory_coverage'] = {label: dict(inventory_count=sum((e['play_type'] == label for e in matches)), overlapping_count=sum((e['play_type'] == label and bool(e['argmax_candidate_ids']) for e in matches))) for label in ['line', 'scrum']}
    result['recall'] = 'Provisional interval overlap with initial independent visual inventory. Pending random checks and visual adjudication of boundary cases.'
    (out / 'initial_inventory_matching.json').write_text(json.dumps(matches, indent=2), encoding='utf-8')
    (out / 'policy_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for filename, events in [('original_candidates.csv', original), ('argmax_candidates.csv', argmax)]:
        with _resolve_model_artifact(out / filename).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['id', 'class', 'start_s', 'end_s', 'confidence', 'visual_label', 'notes'])
            for i, e in enumerate(events, 1):
                w.writerow([i, e['play_type'], e['start_time'], e['end_time'], e['confidence'], '', ''])
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    cap = cv2.VideoCapture(str(P / 'video/jQqN_K0o5_I.mp4'))
    evidence = out / 'precision_review'
    evidence.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    manifest = []
    for first in range(0, len(argmax), 6):
        batch = argmax[first:first + 6]
        sheet = Image.new('RGB', (1920, len(batch) * 224), 'white')
        d = ImageDraw.Draw(sheet)
        for row, e in enumerate(batch):
            a, b = (e['start_time'], e['end_time'])
            peak = (e['peak_window']['start_time'] + e['peak_window']['end_time']) / 2
            times = sorted(set([a + 0.2, a + (b - a) * 0.25, (a + b) / 2, a + (b - a) * 0.75, b - 0.2, peak]))
            idx = first + row + 1
            d.text((4, row * 224 + 2), f"#{idx} {e['play_type']} {a:.1f}-{b:.1f}s", font=font, fill='black')
            for col, t in enumerate(times):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                assert ok
                sheet.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320, 180)), (col * 320, row * 224 + 25))
                d.text((col * 320 + 4, row * 224 + 205), f'{t:.2f}s', font=font, fill='black')
            manifest.append(dict(id=idx, play_type=e['play_type'], start_s=a, end_s=b, times_s=times, sheet=f'sheet_{first // 6 + 1:02}.jpg', reviewed=False))
        sheet.save(evidence / f'sheet_{first // 6 + 1:02}.jpg', quality=94)
    cap.release()
    (evidence / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    gallery = ''.join((f'<h2>Candidatos {i * 6 + 1}–{min((i + 1) * 6, len(argmax))}</h2><img loading="lazy" src="argmax_comparison/precision_review/sheet_{i + 1:02}.jpg">' for i in range((len(argmax) + 5) // 6)))
    doc = f'<!doctype html><html lang="es"><meta charset="utf-8"><title>Segundo partido profesional</title><style>body{{font:16px system-ui;max-width:1400px;margin:auto;padding:24px}}img{{width:100%}}</style><h1>Argentina–All Blacks, Córdoba, 16/08/2025</h1><p>Inferencia completa sobre el segundo partido. Mismo modelo y protocolo, pesos sin modificar.</p><p><b>Precisión y sensibilidad pendientes de revisión visual. Estos son candidatos, no aciertos confirmados.</b></p><pre>{html.escape(json.dumps(result, indent=2))}</pre><p><a href="argmax_comparison/argmax_candidates.csv">Candidatos argmax</a></p>{gallery}</html>'
    (P / 'report.html').write_text(doc, encoding='utf-8')
    (P / 'results/review_ready.json').write_text(json.dumps(dict(inference_complete=True, evidence_ready=True, manual_review_complete=False, report=str(P / 'report.html')), indent=2), encoding='utf-8')
    plan = json.loads(_resolve_model_artifact(P / 'experiment_plan.json').read_text(encoding='utf-8'))
    plan['status'] = 'inference_complete_manual_candidate_review_pending'
    plan['automatic_results'] = result
    (P / 'experiment_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
