"""Current partitions and shared feature preparation for auxiliary trainers."""
from datetime import datetime
from functools import lru_cache
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(os.environ.get('TESINA_DATASET', ROOT/'data/clips_dataset')).resolve()
_RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def training_main(family, argv=None):
    from tools.modelos import main
    return main(['train', '--family', family, *(sys.argv[1:] if argv is None else argv)])


def auxiliary_directory(name, family='M0'):
    base = Path(os.environ.get('TESINA_AUXILIARY_OUTPUT', ROOT/'models'/family/'runs'/('auxiliar_'+_RUN_TAG))).resolve()
    if base.is_relative_to(ROOT.parents[1]) and not base.is_relative_to(ROOT):
        raise ValueError('La salida debe estar fuera del repositorio original')
    return base/name


def current_frames(dataset=None):
    import pandas as pd
    directory = Path(dataset or DATASET).resolve()
    if directory.is_relative_to(ROOT.parents[1]) and not directory.is_relative_to(ROOT):
        raise ValueError('El dataset debe pertenecer al repositorio nuevo')
    frames, owners = [], {}
    for split in ('train', 'validation', 'test'):
        frame = pd.read_csv(directory/(split+'.csv'))
        if not {'video_name', 'tag'}.issubset(frame.columns) or frame.empty:
            raise ValueError(f'Partición vacía o incompatible: {split}')
        for row in frame.to_dict('records'):
            if row['tag'] not in {'line', 'other', 'scrum'}:
                raise ValueError(f'Etiqueta desconocida: {row["tag"]}')
            keys = ['name:'+str(row['video_name'])]
            keys += [key+':'+str(row[key]) for key in ('clip_id', 'path', 'sha256') if row.get(key)]
            for key in keys:
                if key in owners:
                    raise ValueError(f'Clip compartido o asignado más de una vez: {row["video_name"]}')
                owners[key] = split
        frame['split'] = split
        frames.append(frame)
    return tuple(frames)


@lru_cache(maxsize=3)
def _rgb_source(extractor):
    from models.common.workflow import configure
    configure(int(os.environ.get('TESINA_THREADS', '2')))
    from models.common.features import feature_source
    source = feature_source('M0')
    if extractor != 'resnet50':
        import keras
        from models.inference_cache import load_with_revision
        from models.model_catalog import resolve_artifact
        path = resolve_artifact(ROOT/'models/M0/checkpoints'/f'feature_extractor_{extractor}.keras')
        if not path.is_file():
            raise FileNotFoundError(f'Extractor local no disponible: {extractor}')
        source.feature_extractor, source._feature_cache_rgb_revision = load_with_revision(path, lambda p: keras.models.load_model(p, compile=False))
        source.extractor_name = extractor
    return source


def cached_rgb(video_name, extractor='resnet50'):
    from models.common.features import prepare, rgb_inputs
    if Path(video_name).name != video_name:
        raise ValueError('El nombre del clip no debe contener una ruta')
    video = ROOT/'data/videos/clips_224'/video_name
    cache_dir = os.environ.get('TESINA_CACHE_DIR')
    features, _state, _info, _parameters = prepare(_rgb_source(extractor), video, cache_dir=cache_dir)
    return rgb_inputs({'features': features})


def validation_data(label_vocab, extractor='resnet50', sampled_frames=None):
    import numpy as np
    validation = current_frames()[1]
    values, masks, labels = [], [], []
    lookup = {label: i for i, label in enumerate(label_vocab)}
    for row in validation.to_dict('records'):
        if row['tag'] not in lookup:
            continue
        features, mask = cached_rgb(row['video_name'], extractor)
        if sampled_frames is not None:
            from models.M0.train_light_live import build_sparse_temporal_view
            features, mask, _ = build_sparse_temporal_view(features, mask, sampled_frames)
        values.append(features)
        masks.append(mask)
        labels.append(lookup[row['tag']])
    if not labels:
        raise ValueError('Validación no contiene ejemplos de las clases seleccionadas')
    return ([np.asarray(values, np.float32), np.asarray(masks, bool)], np.asarray(labels, np.int32))


def rgb_dataset(frame, label_vocab, extractor='resnet50'):
    import numpy as np
    lookup = {label: i for i, label in enumerate(label_vocab)}
    pairs = [cached_rgb(str(row['video_name']), extractor) for row in frame.to_dict('records')]
    return ((np.asarray([p[0] for p in pairs], np.float32), np.asarray([p[1] for p in pairs], bool)),
            np.asarray([lookup[label] for label in frame['tag']], np.int32))


@lru_cache(maxsize=2)
def _pose_source(pose_model):
    from models.model_catalog import resolve_artifact, temporary_settings
    from models.inference_cache import load_with_revision
    os.environ['YOLO_CONFIG_DIR'] = str(temporary_settings())
    from ultralytics import YOLO
    path = resolve_artifact(ROOT/pose_model)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise FileNotFoundError('Se requieren pesos de pose locales')
    return load_with_revision(path, YOLO)


def cached_pose_arrays(video_name, pose_model, length):
    import hashlib
    import json
    import tempfile
    import numpy as np
    from models.inference_cache import file_digest
    from models.M1.pose_cache import extract_clip
    video = ROOT/'data/videos/clips_224'/video_name
    if Path(video_name).name != video_name:
        raise ValueError('El nombre del clip no debe contener una ruta')
    model, revision = _pose_source(pose_model)
    signature = {'video': file_digest(video), 'pose': revision, 'frames': length,
                 'extract': file_digest(ROOT/'models/M1/pose_cache.py'),
                 'geometry': file_digest(ROOT/'models/M1/geometry_features.py')}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = Path(os.environ.get('TESINA_CACHE_DIR', ROOT/'.runtime/cache/features'))/'geometry'
    path = directory/(key+'.npz')
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    else:
        arrays, _metadata = extract_clip(model, video, length, 640, .25, .2, 30)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    return arrays


def cached_geometry(video_name, pose_model, length):
    arrays = cached_pose_arrays(video_name, pose_model, length)
    return arrays['position_features'], arrays['pose_features'], arrays['frame_mask'].astype(bool)
