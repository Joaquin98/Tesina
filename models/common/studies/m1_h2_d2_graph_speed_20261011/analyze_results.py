"""Aggregate raw experimental results. Refuses to overwrite prior analyses."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import csv
import json
from pathlib import Path
import statistics
import benchmark as isolation
HERE = isolation.HERE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='analysis')
    parser.add_argument('--campaign-prefix', default='')
    args = parser.parse_args()
    if _resolve_model_artifact(HERE / f'results/{args.prefix}.json').exists():
        raise FileExistsError(args.prefix)
    runs = []
    summaries = []
    for path in _model_artifact_glob(HERE / 'results', '*_summary.json', recursive=False):
        summary = json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8'))
        if args.campaign_prefix and (not summary['campaign'].startswith(args.campaign_prefix)):
            continue
        summaries.append(summary)
        for row in summary['runs']:
            runs.append(json.loads(_resolve_model_artifact(HERE / f"results/{row['id']}.json").read_text(encoding='utf-8')))
    scenes = sorted(set((r['video'] for r in runs)))
    output_rows = []
    metrics = ['total_seconds', 'rgb_seconds', 'yolo_seconds', 'recurrent_seconds', 'forest_seconds', 'window_preparation_fusion_seconds', 'geometry_sequence_seconds', 'postprocess_seconds', 'rest_seconds']
    for scene in scenes:
        group = [r for r in runs if r['video'] == scene]
        by_mode = {mode: [r for r in group if r['mode'] == mode] for mode in ['direct', 'graph']}
        row = {'scene': Path(scene).stem, 'video': scene, 'video_seconds': group[0]['result']['metadata']['duration'], 'windows': group[0]['counts']['window_calls'], 'sampled_frames': group[0]['counts']['yolo_calls'], 'repeats_per_mode': len(by_mode['direct']), 'events': len(group[0]['result']['detections'])}
        for metric in metrics:
            for mode in by_mode:
                row[f'{mode}_{metric}'] = statistics.median((r['timings'][metric] for r in by_mode[mode]))
            row[f'{metric}_speedup'] = row[f'direct_{metric}'] / max(1e-12, row[f'graph_{metric}'])
        row['direct_realtime_factor'] = row['direct_total_seconds'] / row['video_seconds']
        row['graph_realtime_factor'] = row['graph_total_seconds'] / row['video_seconds']
        row['direct_extrapolated_80min_hours'] = row['direct_realtime_factor'] * 80 / 60
        row['graph_extrapolated_80min_hours'] = row['graph_realtime_factor'] * 80 / 60
        output_rows.append(row)
    primary = [r for r in output_rows if r['scene'] in ['VF_S01_scrum_492_513', 'VF_S03_line_894_915', 'VF_S05_other_38_59']]
    aggregate = {}
    for name, group in [('original_three_clips', primary), ('continuous_42s', [r for r in output_rows if 'continuous' in r['scene']])]:
        video_seconds = sum((r['video_seconds'] for r in group))
        item = {'video_seconds': video_seconds, 'windows': sum((r['windows'] for r in group)), 'sampled_frames': sum((r['sampled_frames'] for r in group))}
        for metric in metrics:
            for mode in ['direct', 'graph']:
                item[f'{mode}_{metric}'] = sum((r[f'{mode}_{metric}'] for r in group))
            item[f'{metric}_speedup'] = item[f'direct_{metric}'] / max(1e-12, item[f'graph_{metric}'])
        for mode in ['direct', 'graph']:
            item[f'{mode}_realtime_factor'] = item[f'{mode}_total_seconds'] / video_seconds
            item[f'{mode}_extrapolated_80min_hours'] = item[f'{mode}_realtime_factor'] * 80 / 60
            item[f'{mode}_per_window_recurrent_seconds'] = item[f'{mode}_recurrent_seconds'] / item['windows']
            frame_cost = sum((item[f'{mode}_{m}'] for m in ['rgb_seconds', 'yolo_seconds', 'geometry_sequence_seconds'])) / item['sampled_frames']
            window_cost = sum((item[f'{mode}_{m}'] for m in ['recurrent_seconds', 'forest_seconds', 'window_preparation_fusion_seconds'])) / item['windows']
            rest_cost = sum((item[f'{mode}_{m}'] for m in ['rest_seconds', 'postprocess_seconds'])) / video_seconds
            item[f'{mode}_window_count_adjusted_extrapolated_80min_hours'] = (frame_cost * 120000 + window_cost * 4794 + rest_cost * 4800) / 3600
        aggregate[name] = item
    checks = []
    event_numeric_differences = []
    for direct in [r for r in runs if r['mode'] == 'direct']:
        graph = next((r for r in runs if r['id'] == direct['id'].removesuffix('_direct') + '_graph'))
        event_fields = ['play_type', 'start_time', 'end_time', 'frame_start', 'frame_end', 'duration']
        checks.append({'video': direct['video'], 'repeat': direct['repeat'], 'event_count_exact': len(direct['result']['detections']) == len(graph['result']['detections']), 'event_classes_frames_times_exact': all((all((a.get(k) == b.get(k) for k in event_fields)) for a, b in zip(direct['result']['detections'], graph['result']['detections']))), 'window_count_exact': len(direct['window_trace']) == len(graph['window_trace']), 'rgb_masks_exact': all((a['rgb_mask_sha256'] == b['rgb_mask_sha256'] for a, b in zip(direct['window_trace'], graph['window_trace'])))})

        def differences(a, b, path):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in a.keys() & b.keys():
                    differences(a[k], b[k], path + '/' + str(k))
            elif isinstance(a, list) and isinstance(b, list):
                for i, (x, y) in enumerate(zip(a, b)):
                    differences(x, y, path + '/' + str(i))
            elif a != b:
                event_numeric_differences.append({'id': direct['id'], 'path': path, 'direct': a, 'graph': b, 'absolute_difference': abs(a - b) if isinstance(a, (float, int)) and isinstance(b, (float, int)) else None})
        differences(direct['result']['detections'], graph['result']['detections'], 'detections')
        differences(direct['result']['review_detections_7s'], graph['result']['review_detections_7s'], 'review_detections_7s')
    warning_counts = {}
    for log in _model_artifact_glob(HERE / 'logs', '*_console.log', recursive=False):
        active = None
        for line in _resolve_model_artifact(log).read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('BEGIN '):
                active = line.removeprefix('BEGIN ').strip()
                warning_counts.setdefault(active, 0)
            elif line.startswith('END '):
                active = None
            elif active and 'parallel.py:144: UserWarning' in line:
                warning_counts[active] += 1
    result = {'scenes': output_rows, 'aggregates': aggregate, 'additional_checks': checks, 'event_numeric_differences': event_numeric_differences, 'sklearn_parallel_warning_counts': warning_counts, 'equivalence': [e for s in summaries for e in s['equivalence']], 'integrity': [{k: s[k] for k in ['campaign', 'reference_content_unchanged', 'existing_files_changed', 'unexpected_new_files_outside_experiment', 'graph_tracing_counts']} for s in summaries]}
    isolation.save(f'results/{args.prefix}.json', result)
    with _resolve_model_artifact(HERE / f'results/{args.prefix}_timing_summary.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'figures').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    main_rows = primary + [r for r in output_rows if 'continuous' in r['scene']]
    labels = ['Scrum 21 s', 'Line 21 s', 'Other 21 s', 'Continuo 42 s']
    x = list(range(len(main_rows)))
    axes[0].bar([v - 0.18 for v in x], [r['direct_realtime_factor'] for r in main_rows], 0.36, label='Llamada directa', color='#8793a0')
    axes[0].bar([v + 0.18 for v in x], [r['graph_realtime_factor'] for r in main_rows], 0.36, label='tf.function', color='#226a91')
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('Segundos de cómputo / segundo de video')
    axes[0].set_title('Tiempo completo, modelos calientes')
    axes[0].legend()
    groups = ['rgb_seconds', 'yolo_seconds', 'recurrent_seconds', 'forest_seconds']
    g = aggregate['original_three_clips']
    vals = [100 * g['graph_' + m] / g['graph_total_seconds'] for m in groups]
    vals.append(100 - sum(vals))
    axes[1].barh(['ResNet', 'YOLO y conversión', 'Redes recurrentes', 'Bosques', 'Resto'], vals, color=['#5086a0', '#226a91', '#7fa6b3', '#afc6ce', '#ced9dd'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('% del tiempo completo en grafo')
    axes[1].set_title('Qué limita la variante en grafo')
    for i, v in enumerate(vals):
        axes[1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)
    axes[1].set_xlim(0, max(vals) + 9)
    fig.savefig(HERE / f'figures/{args.prefix}_timing_comparison.png', dpi=180)
    fig.savefig(HERE / f'figures/{args.prefix}_timing_comparison.svg')
    print(json.dumps(aggregate, indent=2))
if __name__ == '__main__':
    main()
