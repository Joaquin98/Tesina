"""M0 training and adaptation for the comparative protocol."""
from __future__ import annotations
from models.training_common import *

def _configure_m0_trainable(model, level: str) -> None:
    trainable = {'H0': {'dense_1'}, 'H1': {'bidirectional_1', 'dense', 'dense_1'}, 'H2': {'bidirectional', 'bidirectional_1', 'dense', 'dense_1'}}[level]
    for layer in model.layers:
        layer.trainable = layer.name in trainable

def fit_m0(train: pd.DataFrame, development: pd.DataFrame, *, level: str, seed: int, output: Path, epochs: int, patience: int, batch_size: int, base_path: Path=M0_PATH):
    import keras
    seed_everything(seed)
    model = keras.models.load_model(_resolve_model_artifact(base_path), compile=False)
    if level != 'B':
        _configure_m0_trainable(model, level)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        train_sequence = ResNetCacheSequence(train, batch_size=batch_size, shuffle=True, seed=seed)
        dev_sequence = ResNetCacheSequence(development, batch_size=batch_size, shuffle=False, seed=seed)
        history = model.fit(train_sequence, validation_data=dev_sequence, epochs=epochs, class_weight=_class_weights(labels_for(train)), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], verbose=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
        output.with_suffix('.history.json').write_text(json.dumps(history.history, indent=2), encoding='utf-8')
    return model
