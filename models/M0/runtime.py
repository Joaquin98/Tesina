"""Train and predict the current RGB sequence model from local clips."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.model_catalog import resolve_artifact
HERE = Path(__file__).resolve().parent
LABELS = ['line', 'other', 'scrum']
EXTRACTOR = HERE/'checkpoints/feature_extractor_resnet50.keras'
REFERENCE = HERE/'checkpoints/best_model/best_model.keras'


def encode_frames(frames, encoder, chunk_size=8):
    if not len(frames):
        raise ValueError('El video no contiene fotogramas decodificables')
    length = min(200, len(frames))
    features = np.zeros((200, 2048), np.float32)
    for start in range(0, length, chunk_size):
        end = min(start+chunk_size, length)
        value = np.asarray(encoder(np.asarray(frames[start:end], np.float32), training=False))
        if value.shape != (end-start, 2048) or not np.isfinite(value).all():
            raise ValueError('Salida del extractor no válida')
        features[start:end] = value
    return features, np.arange(200) < length


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if any(row['tag'] not in LABELS for row in rows):
        raise ValueError('Etiquetas no reconocidas')
    return rows


def split_unique(rows, *, seed, validation_fraction):
    if len({r['video_name'] for r in rows}) != len(rows):
        raise ValueError('Separar videos únicos antes de sobremuestrear')
    from sklearn.model_selection import train_test_split
    return train_test_split(rows, test_size=validation_fraction, random_state=seed,
                            stratify=[row['tag'] for row in rows])


def segment_frames(path, start=0., seconds=None, frame_limit=200):
    import cv2
    sys.path.insert(0, str(ROOT))
    from models.common.spatial import normalize_training_frame
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'No se puede abrir: {path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError('FPS no válido')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total/fps
        end = duration if seconds is None else min(duration, start+seconds)
        count = min(frame_limit, max(0, int(np.floor((end-start)*25+1e-6))))
        indices = np.clip(np.rint((start+np.arange(count)/25)*fps).astype(int), 0, max(0, total-1))
        if not len(indices):
            raise ValueError('Intervalo sin fotogramas')
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        cursor = int(indices[0])-1
        frames = []
        for index in indices:
            while cursor < index:
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Video incompleto: {path}')
                cursor += 1
            # Corpus clips already contain the training crop.
            normalized = image if image.shape[:2] == (224, 224) else normalize_training_frame(image, target_size=224)
            if normalized is None:
                raise ValueError('No se puede aplicar el recorte de entrenamiento')
            frames.append(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
        return np.asarray(frames), duration
    finally:
        cap.release()


def load_models(sequence_path=REFERENCE):
    import keras
    return (keras.models.load_model(resolve_artifact(EXTRACTOR), compile=False),
            keras.models.load_model(resolve_artifact(sequence_path), compile=False))


def build_sequence_model():
    import keras
    features = keras.Input((200, 2048), name='features')
    mask = keras.Input((200,), dtype='bool', name='mask')
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(features, mask=mask)
    x = keras.layers.Bidirectional(keras.layers.GRU(32))(x)
    x = keras.layers.Dropout(.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model([features, mask], output, name='resnet50_bigru_sin_cache')
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def training_partitions(directory):
    """Consume saved disjoint partitions; never split oversampled rows again."""
    from tools.dataset import local_path, read_csv, verify
    directory = local_path(directory)
    verify(directory)
    return tuple(read_csv(directory/f'{name}.csv') for name in
                 ['balanced_train', 'balanced_validation', 'test'])


def train(args):
    from models.common.workflow import train_run
    import json
    result = train_run('M0', dataset=args.dataset, output=args.output,
                       epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False))
    return result


def predict(args):
    if args.architecture == 'm1':
        import keras
        from models.M1.predictor import Head
        encoder = keras.models.load_model(resolve_artifact(EXTRACTOR), compile=False)
        head = Head()
    else:
        encoder, model = load_models(args.model)
    start = 0.
    writer = csv.writer(sys.stdout)
    writer.writerow(['start_s', 'end_s', 'prediction', 'p_line', 'p_other', 'p_scrum'])
    while True:
        frames, duration = segment_frames(args.video, start, args.segment_seconds)
        features, mask = encode_frames(frames, encoder)
        p = head.predict(features, mask, frames) if args.architecture == 'm1' else np.asarray(model([features[None], mask[None]], training=False))[0]
        if not np.isfinite(p).all() or not np.isclose(p.sum(), 1, atol=1e-5):
            raise ValueError('Probabilidades no válidas')
        end = min(start+args.segment_seconds, duration)
        writer.writerow([start, end, LABELS[int(p.argmax())], *p.tolist()])
        if end >= duration-1/25:
            break
        start = end


def check():
    encoder, model = load_models()
    row = read_rows(HERE/'test.csv')[0]
    frames, _ = segment_frames(ROOT/'data/videos/clips_224'/row['video_name'], frame_limit=4)
    features, mask = encode_frames(frames, encoder, chunk_size=2)
    p = np.asarray(model([features[None], mask[None]], training=False))[0]
    assert features.shape == (200, 2048) and int(mask.sum()) == 4
    assert p.shape == (3,) and np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-5)
    # Verify the training graph with synthetic features, without touching a checkpoint.
    fresh = build_sequence_model()
    loss = fresh.train_on_batch([features[None], mask[None]], np.asarray([0], np.int32))
    assert np.isfinite(loss).all()
    print(json.dumps(dict(status='ok', feature_cache=False, labels=LABELS,
                          extractor_shape=list(features.shape), probabilities=p.tolist(),
                          training_batch_finite=True, test_video=row['video_name'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='Probar extracción, inferencia y un batch sintético de entrenamiento')
    fitting = commands.add_parser('train', help='Entrenar M0 con particiones explícitas y caché local')
    fitting.add_argument('--epochs', type=int, default=100)
    fitting.add_argument('--batch-size', type=int, default=16)
    fitting.add_argument('--seed', type=int, default=42)
    fitting.add_argument('--dataset', type=Path, default=ROOT/'data/clips_dataset',
                         help='Versión con particiones explícitas y rutas locales, creada por scripts/data_prep/create_balanced_splits.py')
    fitting.add_argument('--output', type=Path, default=None)
    prediction = commands.add_parser('predict', help='Predecir ventanas sucesivas sin guardar características')
    prediction.add_argument('--video', type=Path, required=True)
    prediction.add_argument('--model', type=Path, default=REFERENCE, help='Checkpoint para --architecture rgb')
    prediction.add_argument('--architecture', choices=['m1','rgb'], default='rgb', help='M1 RGB+pose o referencia RGB BiGRU')
    prediction.add_argument('--segment-seconds', type=float, default=7.)
    args = parser.parse_args()
    if args.command == 'train' and (args.epochs < 1 or args.batch_size < 1):
        parser.error('epochs y batch-size deben ser positivos')
    if args.command == 'predict' and not 0 < args.segment_seconds <= 8:
        parser.error('segment-seconds debe ser mayor que cero y no superar ocho segundos')
    if args.command == 'predict' and args.architecture == 'm1' and args.model != REFERENCE:
        parser.error('--model sólo se usa con --architecture rgb; M1 carga su par de pesos verificados')
    {'check': lambda _: check(), 'train': train, 'predict': predict}[args.command](args)


if __name__ == '__main__':
    main()
