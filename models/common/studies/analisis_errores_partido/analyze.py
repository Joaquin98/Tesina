import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
OUT = experiment_directory(__file__)
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events, temporal_metrics_for_class

def digest(path):
    return hashlib.sha256(_resolve_model_artifact(Path(path)).read_bytes()).hexdigest()

def main():
    manifest_path = ROOT / 'data/final_tests_feature_cache_20260826/outputs/final_test_feature_manifest.csv'
    manifest = pd.read_csv(_resolve_model_artifact(manifest_path))
    assert manifest.dataset_id.is_unique
    sources = [{'path': str(manifest_path), 'sha256': digest(manifest_path)}]
    for name in ['metrics.py', 'protocol.py']:
        p = ROOT / 'data/final_model_comparison_20260827' / name
        sources.append({'path': str(p), 'sha256': digest(p)})
    all_segments, all_events, summaries = ([], [], [])
    for model in ['M1', 'M3']:
        frames = []
        for fold in range(1, 7):
            paths = list(_model_artifact_glob(ROOT / f'data/final_model_comparison_20260827/outputs/runs/PV-{model}-H2-D2-PV{fold:02}-S42', 'attempt_*/predictions.csv', recursive=False))
            assert len(paths) == 1, paths
            p = paths[0]
            sources.append({'path': str(p), 'sha256': digest(p)})
            f = pd.read_csv(_resolve_model_artifact(p)).merge(manifest[['dataset_id', 'source_video', 'section_file']], left_on='block_id', right_on='dataset_id', validate='one_to_one')
            f['group_id'] = f.test_group
            f['model'] = model
            assert len(f) == len(pd.read_csv(_resolve_model_artifact(p)))
            run = json.loads(_resolve_model_artifact(p.parent / 'run.json').read_text())
            sources.append({'path': str(p.parent / 'run.json'), 'sha256': digest(p.parent / 'run.json')})
            metrics_saved = json.loads(_resolve_model_artifact(p.parent / 'metrics.json').read_text())
            sources.append({'path': str(p.parent / 'metrics.json'), 'sha256': digest(p.parent / 'metrics.json')})
            for label in ['line', 'scrum']:
                true = events_from_blocks(f, label, prediction_column='y_true', probability_column=f'p_{label}')
                pred = events_from_blocks(f, label, prediction_column='y_final', probability_column=f'p_{label}')
                matches = match_events(true, pred)
                matched = {j: (i, d, w) for i, j, d, w in matches}
                m = temporal_metrics_for_class(f, label)
                assert abs(m['weighted_f1'] - metrics_saved['temporal'][label]['weighted_f1']) < 1e-10
                assert abs(m['far_false_positives_per_hour'] - metrics_saved['temporal'][label]['far_false_positives_per_hour']) < 1e-10
                assert m['duplicates'] == metrics_saved['temporal'][label]['duplicates']
                nfar = 0
                for j, e in enumerate(pred):
                    rows = f[(f.section == e.section) & f.block_index.isin(e.blocks)]
                    candidates = [(i, abs(t.anchor - e.anchor)) for i, t in enumerate(true) if t.section == e.section]
                    nearest = min(candidates, key=lambda x: x[1]) if candidates else (None, None)
                    status = 'matched' if j in matched else 'near_unmatched' if nearest[1] is not None and nearest[1] <= 35 else 'far_fp'
                    nfar += status == 'far_fp'
                    eid = f'{model}_V{fold}_{e.section}_{label}_{min(e.blocks):04d}'
                    i, d, w = matched.get(j, (None, None, None))
                    er = dict(event_id=eid, model=model, match=f'V{fold}', section=e.section, prediction=label, start_s=e.start, end_s=e.end, anchor_s=e.anchor, n_segments=len(e.blocks), n_other=int((rows.y_true == 'other').sum()), n_wrong=int((rows.y_true != label).sum()), n_correct=int((rows.y_true == label).sum()), temporal_status=status, matched_truth_start_s=true[i].start if i is not None else None, matched_truth_end_s=true[i].end if i is not None else None, matched_truth_anchor_s=true[i].anchor if i is not None else None, match_distance_s=d, match_weight=w, nearest_same_class_distance_s=nearest[1], source_video=rows.source_video.iloc[0], block_ids='|'.join(rows.block_id))
                    all_events.append(er)
                    f.loc[rows.index, 'analysis_event_id'] = eid
                    f.loc[rows.index, 'temporal_status'] = status
                summaries.append(dict(model=model, match=f'V{fold}', label=label, **m, far_fp=nfar, unmatched_truth=len(true) - len(matches), threshold=run['thresholds'][label]))
            frames.append(f)
        f = pd.concat(frames, ignore_index=True)
        assert len(f) == 4714 and f.block_id.is_unique
        all_segments.append(f)
        for decision in ['y_argmax', 'y_final']:
            c = pd.crosstab(f.y_true, f[decision]).reindex(index=['line', 'other', 'scrum'], columns=['line', 'other', 'scrum'], fill_value=0)
            c.to_csv(OUT / f'confusion_{model}_{decision}.csv')
    seg = pd.concat(all_segments, ignore_index=True)
    seg.to_csv(OUT / 'segments_all.csv', index=False)
    ev = pd.DataFrame(all_events)
    ev.to_csv(OUT / 'events_all.csv', index=False)
    pd.DataFrame(summaries).to_json(OUT / 'temporal_metrics.json', orient='records', indent=2)
    pd.DataFrame(summaries).drop(columns=['f1_at_seconds', 'f1_at_iou']).to_csv(OUT / 'temporal_metrics.csv', index=False)
    (OUT / 'source_hashes.json').write_text(json.dumps(sources, indent=2), encoding='utf8')
    candidates = ev[(ev.model == 'M1') & (ev.n_other > 0)].copy().sort_values(['match', 'section', 'start_s'])
    candidates.to_csv(OUT / 'review_population.csv', index=False)
    print('M1 error segments:', (seg[seg.model == 'M1'].y_true != seg[seg.model == 'M1'].y_final).sum())
    print('Review population', len(candidates))
    print(candidates.groupby(['match', 'prediction']).agg(events=('event_id', 'size'), segments=('n_other', 'sum')).to_string())
    print('All segments confusion:')
    print(pd.crosstab(seg[seg.model == 'M1'].y_true, seg[seg.model == 'M1'].y_final))
if __name__ == '__main__':
    main()
