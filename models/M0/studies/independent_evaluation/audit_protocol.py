"""Audit historical data use and candidate external material.

Run from the repository root:
    python models/M0/studies/independent_evaluation/audit_protocol.py --deep
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M0.studies.independent_evaluation.protocol import BALANCED_TRAIN, CLIP_ROOT, DATA_DIR, EXPERIMENT_DIR, LABELS, MODEL_DIR, OUTPUT_DIR, REPO_ROOT, SOURCE_TEST, SOURCE_TRAIN, class_counts, infer_source_group, read_manifest, sha256_file, source_manifest, write_json

def keras_validation_overlap() -> tuple[pd.DataFrame, dict[str, object]]:
    balanced = read_manifest(BALANCED_TRAIN)
    cut = int(len(balanced) * 0.7)
    internal_train = balanced.iloc[:cut].copy()
    internal_val = balanced.iloc[cut:].copy()
    overlap = internal_train[['video_name', 'tag']].drop_duplicates().merge(internal_val[['video_name', 'tag']].drop_duplicates(), on=['video_name', 'tag']).sort_values(['tag', 'video_name'])
    return (overlap, {'balanced_rows': int(len(balanced)), 'balanced_unique_clips': int(balanced.video_name.nunique()), 'keras_internal_train_rows': int(len(internal_train)), 'keras_internal_validation_rows': int(len(internal_val)), 'overlap_unique_clips': int(overlap.video_name.nunique()), 'overlap_by_class': {label: int(overlap.loc[overlap.tag == label, 'video_name'].nunique()) for label in LABELS}})

def historical_results_audit() -> dict[str, object]:
    grid_path = MODEL_DIR / 'results' / 'grid_search_results.csv'
    grid = pd.read_csv(_resolve_model_artifact(grid_path))
    metrics = [col for col in grid.columns if col.startswith('test_') or col.startswith('recall_')]
    metrics.extend(['val_f1_macro (computed from test_labels)', 'val_f1_weighted (computed from test_labels)'])
    source_files = [_resolve_model_artifact(MODEL_DIR / 'grid_search.py'), _resolve_model_artifact(MODEL_DIR / 'train_best_focused.py'), _resolve_model_artifact(MODEL_DIR / 'analyze_tuning.py')]
    evidence = {}
    for path in source_files:
        text = _resolve_model_artifact(path).read_text(encoding='utf-8', errors='replace') if _resolve_model_artifact(path).exists() else ''
        evidence[str(path.relative_to(REPO_ROOT))] = {'mentions_test_accuracy': 'test_accuracy' in text, 'sorts_or_tiebreaks_with_test': 'sort_values' in text and 'test_accuracy' in text}
    return {'grid_rows': int(len(grid)), 'expected_factorial_grid': 576, 'test_metrics_recorded_for_each_configuration': metrics, 'grid_file_current_order': 'not guaranteed to be the original selection order', 'selection_evidence': evidence, 'conclusion': 'The 817-example test partition was evaluated for every grid configuration. The columns named val_f1_macro and val_f1_weighted were actually computed from test_features/test_labels, and val_f1_macro was the primary sort key. In addition, test_accuracy was later used in focused ranking/tie-breaking.'}

def youtube_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roots = [DATA_DIR / 'raw' / 'youtube' / 'Pablo Granitto', DATA_DIR / 'raw' / 'youtube' / 'Juan Manuel Arriazu', DATA_DIR / 'raw' / 'web_downloads' / 'youtube_external_tests']
    for root in roots:
        if not _resolve_model_artifact(root).exists():
            continue
        for path in _model_artifact_glob(root, '*.mp4', recursive=False):
            video_id = path.stem.rsplit('_', 1)[-1]
            rows.append({'path': str(path.relative_to(REPO_ROOT)), 'collection': root.name, 'youtube_id': video_id, 'bytes': path.stat().st_size, 'modified_utc': path.stat().st_mtime, 'historical_role': 'candidate_external_unlabelled' if root.name == 'youtube_external_tests' else 'interactive_tests_and_source_for_human_review'})
    return pd.DataFrame(rows)

def human_review_inventory() -> tuple[pd.DataFrame, dict[str, object]]:
    manifests = sorted(_model_artifact_glob(DATA_DIR / 'active_learning_dataset', '*.csv', recursive=True))
    rows: list[dict[str, str]] = []
    for path in manifests:
        try:
            df = pd.read_csv(_resolve_model_artifact(path))
        except Exception:
            continue
        path_col = next((c for c in ('video_name', 'clip_path', 'exported_clip_path') if c in df), None)
        if not path_col:
            continue
        tag_col = next((c for c in ('tag', 'corrected_label', 'label') if c in df), None)
        for _, row in df.iterrows():
            value = str(row.get(path_col, '')).strip()
            if not value or value.lower() == 'nan':
                continue
            rows.append({'manifest': str(path.relative_to(REPO_ROOT)), 'clip': value, 'label': str(row.get(tag_col, '')) if tag_col else ''})
    out = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=['manifest', 'clip', 'label'])
    summary = {'csv_manifests_scanned': len(manifests), 'distinct_manifest_clip_references': int(out['clip'].nunique()) if len(out) else 0, 'retraining_manifest_present': _resolve_model_artifact(DATA_DIR / 'active_learning_dataset' / 'retraining_set_860' / 'train.csv').exists(), 'known_retraining_outputs': ['models/M0/results/human_review_retrain', 'models/M0/results/human_review_retrain_860']}
    return (out, summary)

def dhash_frame(frame: np.ndarray) -> int:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = value << 1 | int(bit)
    return value

def sampled_hashes(path: Path, samples: int, middle_only: bool=False) -> list[int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        indices = [total // 2] if middle_only else np.linspace(0, total - 1, samples, dtype=int).tolist()
        values = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                values.append(dhash_frame(frame))
        return values
    finally:
        cap.release()

def deep_external_overlap(all_clips: pd.DataFrame, yt: pd.DataFrame) -> dict[str, object]:
    """Conservative exact and sampled visual comparison for external raw videos."""
    external = yt[yt.historical_role == 'candidate_external_unlabelled'].copy()
    dev_hashes: dict[str, list[str]] = {}
    missing = 0
    clip_visual: list[tuple[str, int]] = []
    for name in all_clips.video_name:
        path = CLIP_ROOT / name
        if not _resolve_model_artifact(path).exists():
            missing += 1
            continue
        digest = sha256_file(path)
        dev_hashes.setdefault(digest, []).append(name)
        hashes = sampled_hashes(path, 1, middle_only=True)
        if hashes:
            clip_visual.append((name, hashes[0]))
    exact = []
    near = []
    for rel in external.path:
        path = REPO_ROOT / rel
        digest = sha256_file(path)
        if digest in dev_hashes:
            exact.append({'external': rel, 'clips': dev_hashes[digest]})
        for ext_hash in sampled_hashes(path, 40):
            matches = [name for name, value in clip_visual if (ext_hash ^ value).bit_count() <= 4]
            if matches:
                near.append({'external': rel, 'matching_middle_frames': matches[:20]})
    duplicate_development_hashes = [names for names in dev_hashes.values() if len(names) > 1]
    return {'development_files_hashed': sum((len(v) for v in dev_hashes.values())), 'development_files_missing': missing, 'exact_duplicate_sets_inside_development': duplicate_development_hashes, 'exact_external_matches': exact, 'sampled_visual_matches_hamming_le_4': near, 'visual_method_limit': 'Middle-frame dHash for each labelled clip versus 40 uniformly sampled frames per external full video. Absence is evidence against obvious duplication, not a proof that no short segment anywhere in the full videos is related.'}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--deep', action='store_true', help='Hash files and compare sampled video frames')
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_clips = source_manifest()
    overlap, overlap_summary = keras_validation_overlap()
    overlap.to_csv(OUTPUT_DIR / 'historical_internal_overlap_92.csv', index=False)
    balanced_historical = read_manifest(BALANCED_TRAIN)
    historical_cut = int(len(balanced_historical) * 0.7)
    balanced_historical.iloc[:historical_cut].to_csv(OUTPUT_DIR / 'historical_internal_train_rows.csv', index=False)
    balanced_historical.iloc[historical_cut:].to_csv(OUTPUT_DIR / 'historical_internal_validation_rows.csv', index=False)
    all_clips.to_csv(OUTPUT_DIR / 'historical_clip_master.csv', index=False)
    yt = youtube_inventory()
    yt.to_csv(OUTPUT_DIR / 'youtube_video_inventory.csv', index=False)
    review, review_summary = human_review_inventory()
    review.to_csv(OUTPUT_DIR / 'human_review_clip_inventory.csv', index=False)
    original_train = read_manifest(SOURCE_TRAIN)
    original_test = read_manifest(SOURCE_TEST)
    name_overlap = sorted(set(original_train.video_name) & set(original_test.video_name))
    group_overlap = sorted(set(original_train.video_name.map(infer_source_group)) & set(original_test.video_name.map(infer_source_group)))
    external_ids = set(yt.loc[yt.historical_role == 'candidate_external_unlabelled', 'youtube_id'])
    used_ids = set(yt.loc[yt.historical_role != 'candidate_external_unlabelled', 'youtube_id'])
    audit = {'protocol': 'historical_protocol_audit', 'source_manifest': {'development_rows': int(len(original_train)), 'development_unique_clips': int(original_train.video_name.nunique()), 'test_rows': int(len(original_test)), 'test_unique_clips': int(original_test.video_name.nunique()), 'combined_unique_clips': int(all_clips.video_name.nunique()), 'class_counts': class_counts(all_clips), 'source_groups_inferred': int(all_clips.source_group.nunique()), 'missing_clip_files': int((~all_clips.clip_exists).sum()), 'train_test_name_overlap': len(name_overlap), 'train_test_group_overlap': len(group_overlap), 'train_test_group_examples': group_overlap[:30]}, 'keras_validation_leakage': overlap_summary, 'configuration_selection': historical_results_audit(), 'youtube': {'interactive_or_review_source_videos': int((yt.historical_role != 'candidate_external_unlabelled').sum()), 'candidate_external_full_videos': int((yt.historical_role == 'candidate_external_unlabelled').sum()), 'candidate_ids_also_in_interactive_collection': sorted(external_ids & used_ids), 'assessment': 'Three source videos have distinct YouTube IDs and no repository reference outside their external-tests folder, but they contain no independent human ground-truth clip manifest. They are candidates for future annotation, not a usable external evaluation set today.'}, 'human_review': review_summary, 'independence_decision': {'fully_independent_labelled_set_available': False, 'independent_unlabelled_source_candidates_available': True, 'chosen_strategy': 'nested_grouped_cross_validation', 'reason': 'All 4082 labelled clips belonged to historical development or the repeatedly queried 817-example test set. The only provenance-new material is unlabelled, so it cannot produce accuracy/F1 without contaminating ground truth with model predictions.'}}
    if args.deep:
        audit['deep_overlap'] = deep_external_overlap(all_clips, yt)
    else:
        previous_path = OUTPUT_DIR / 'audit_protocol.json'
        previous_deep = None
        if _resolve_model_artifact(previous_path).exists():
            try:
                previous_deep = json.loads(_resolve_model_artifact(previous_path).read_text(encoding='utf-8')).get('deep_overlap')
            except Exception:
                previous_deep = None
        audit['deep_overlap'] = previous_deep if previous_deep and 'development_files_hashed' in previous_deep else {'status': 'not_run', 'command': 'audit_protocol.py --deep'}
    write_json(OUTPUT_DIR / 'audit_protocol.json', audit)
    lines = ['# AuditorÃ­a del protocolo histÃ³rico', '', f'- Clips etiquetados Ãºnicos: {all_clips.video_name.nunique()} ({original_train.video_name.nunique()} de desarrollo y {original_test.video_name.nunique()} de prueba).', f'- Solapamiento nominal desarrollo/prueba: {len(name_overlap)} clips. Solapamiento por grupo fuente inferido: {len(group_overlap)} grupos.', f"- La tabla balanceada tiene {overlap_summary['balanced_rows']} filas y {overlap_summary['balanced_unique_clips']} clips Ãºnicos.", f"- La validaciÃ³n interna histÃ³rica repite {overlap_summary['overlap_unique_clips']} clips ({overlap_summary['overlap_by_class']}).", f"- La grilla contiene {audit['configuration_selection']['grid_rows']} resultados y registra mÃ©tricas de prueba para cada configuraciÃ³n. La columna `val_f1_macro` se calculÃ³ con `test_labels` y fue la clave principal de ordenamiento; programas posteriores tambiÃ©n usaron `test_accuracy` para ordenar o desempatar candidatos.", f"- Se inventariaron {(yt.historical_role != 'candidate_external_unlabelled').sum()} videos de YouTube usados en herramientas/pruebas interactivas o como fuente de revisiÃ³n humana.", f"- Se hallaron {(yt.historical_role == 'candidate_external_unlabelled').sum()} videos fuente nuevos, pero no tienen etiquetas humanas independientes por clip.", '- ConclusiÃ³n: no existe hoy un conjunto externo etiquetado completamente independiente. Se adopta evaluaciÃ³n cruzada anidada y agrupada.']
    (OUTPUT_DIR / 'AUDIT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(audit['independence_decision'], ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
