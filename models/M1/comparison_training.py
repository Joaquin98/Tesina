"""M1 training and adaptation for the comparative protocol."""
from __future__ import annotations
from models.training_common import *
from models.M0.comparison_training import fit_m0

def build_dual_model(base_path: Path, *, level: str):
    import keras
    rgb = keras.Input((200, 2048), name='rgb')
    pose = keras.Input((200, 62), name='pose')
    mask = keras.Input((200,), dtype='bool', name='sequence_mask')
    rgb_sequence = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='rgb_bigru_1')(rgb, mask=mask)
    rgb_embedding = keras.layers.Bidirectional(keras.layers.GRU(32), name='rgb_bigru_2')(rgb_sequence, mask=mask)
    pose_values = keras.layers.LayerNormalization(name='pose_norm')(pose)
    pose_sequence = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=True), name='pose_bigru_1')(pose_values, mask=mask)
    pose_embedding = keras.layers.Bidirectional(keras.layers.GRU(32), name='pose_bigru_2')(pose_sequence, mask=mask)
    fused = keras.layers.Concatenate(name='fusion_concat')([rgb_embedding, pose_embedding])
    fused = keras.layers.Dropout(0.3, name='fusion_dropout')(fused)
    fused = keras.layers.Dense(64, activation='relu', name='fusion_dense')(fused)
    output = keras.layers.Dense(3, activation='softmax', name='class_output')(fused)
    model = keras.Model({'rgb': rgb, 'pose': pose, 'sequence_mask': mask}, output, name='m1_100frames')
    source = keras.models.load_model(_resolve_model_artifact(base_path), compile=False)
    for source_name, target_name in (('bidirectional', 'rgb_bigru_1'), ('bidirectional_1', 'rgb_bigru_2')):
        model.get_layer(target_name).set_weights(source.get_layer(source_name).get_weights())
    source_dense = source.get_layer('dense').get_weights()
    target_dense = model.get_layer('fusion_dense').get_weights()
    expanded = np.zeros_like(target_dense[0])
    expanded[:source_dense[0].shape[0], :] = source_dense[0]
    model.get_layer('fusion_dense').set_weights([expanded, source_dense[1]])
    model.get_layer('class_output').set_weights(source.get_layer('dense_1').get_weights())
    if level == 'H1':
        for name in ('rgb_bigru_1', 'rgb_bigru_2'):
            model.get_layer(name).trainable = False
    return model

def fit_m1_dual(train, development, *, level, seed, output, epochs, patience, batch_size):
    import keras
    seed_everything(seed)
    model = build_dual_model(M0_PATH, level=level)
    model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(ResNetCacheSequence(train, batch_size=batch_size, shuffle=True, seed=seed, dual=True), validation_data=ResNetCacheSequence(development, batch_size=batch_size, shuffle=False, seed=seed, dual=True), epochs=epochs, class_weight=_class_weights(labels_for(train)), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)], verbose=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    output.with_suffix('.history.json').write_text(json.dumps(history.history, indent=2), encoding='utf-8')
    return model

class M1ExactSequence(keras.utils.Sequence):

    def __init__(self, frame: pd.DataFrame, *, batch_size: int, pose: bool):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.pose = bool(pose)

    def __len__(self):
        return int(np.ceil(len(self.frame) / self.batch_size))

    def __getitem__(self, batch_index):
        rows = self.frame.iloc[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        rgb, masks, pose = ([], [], [])
        for row in rows.itertuples(index=False):
            features, mask, _length = load_resnet(row)
            rgb.append(features)
            masks.append(mask)
            if self.pose:
                raw = load_m1_pose32(row)
                pose_values = align_sequence(raw['pose_features'], 200)
                pose_mask = align_sequence(raw['frame_mask'].astype(np.float32)[:, None], 200)[:, 0]
                pose.append(np.concatenate([pose_values, pose_mask[:, None]], axis=1))
        rgb = np.asarray(rgb, dtype=np.float32)
        masks = np.asarray(masks, dtype=bool)
        if self.pose:
            inputs = {'rgb': rgb, 'pose': np.asarray(pose, dtype=np.float32), 'sequence_mask': masks}
        else:
            inputs = {'rgb': rgb, 'rgb_mask': masks}
        return inputs

def _m1_rgb_model():
    rgb = keras.Input((200, 2048), name='rgb')
    mask = keras.Input((200,), dtype='bool', name='rgb_mask')
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='rgb_bigru_1')(rgb, mask=mask)
    x = keras.layers.Bidirectional(keras.layers.GRU(32), name='rgb_bigru_2')(x, mask=mask)
    x = keras.layers.Dropout(0.3, name='fusion_dropout')(x)
    x = keras.layers.Dense(64, activation='relu', name='fusion_dense')(x)
    output = keras.layers.Dense(3, activation='softmax', name='class_output')(x)
    return keras.Model({'rgb': rgb, 'rgb_mask': mask}, output)

def _m1_pose_model():
    rgb = keras.Input((200, 2048), name='rgb')
    pose = keras.Input((200, 31), name='pose')
    mask = keras.Input((200,), dtype='bool', name='sequence_mask')
    x = keras.layers.Concatenate(name='same_rnn_input')([rgb, pose])
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')(x, mask=mask)
    x = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')(x, mask=mask)
    x = keras.layers.Dropout(0.3, name='same_dropout')(x)
    x = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(x)
    output = keras.layers.Dense(3, activation='softmax', name='same_class_output')(x)
    return keras.Model({'rgb': rgb, 'pose': pose, 'sequence_mask': mask}, output)

def _m1_geometry_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([geometry_statistics(load_m1_pose32(row)) for row in frame.itertuples(index=False)], dtype=np.float32)

def _predict_m1_exact(train, development, test, *, level, seed, run_dir, batch_size) -> PredictionBundle:
    manifest_path = M1_EXPORT / 'manifest.json'
    manifest = json.loads(_resolve_model_artifact(manifest_path).read_text(encoding='utf-8'))
    geometry_train = _m1_geometry_matrix(train) if level == 'H0' else None
    geometry_dev = _m1_geometry_matrix(development)
    geometry_test = _m1_geometry_matrix(test)
    dev_seed, test_seed = ([], [])
    dev_rgb, dev_pose, dev_geometry = ([], [], [])
    test_rgb, test_pose, test_geometry = ([], [], [])
    artifacts = [manifest_path]
    class_weights = np.asarray([0.8, 1.0, 0.9], dtype=np.float64)
    for run in manifest['runs']:
        original_seed = int(run['seed'])
        rgb_path = M1_ROOT / 'm0' / f'seed_{original_seed}' / 'best.weights.h5'
        pose_path = M1_ROOT / 'same_rnn' / 'pose_concat' / f'seed_{original_seed}' / 'best.weights.h5'
        rgb_model = _m1_rgb_model()
        rgb_model.load_weights(_resolve_model_artifact(rgb_path))
        pose_model = _m1_pose_model()
        pose_model.load_weights(_resolve_model_artifact(pose_path))
        rgb_dev = normalize_probabilities(rgb_model.predict(M1ExactSequence(development, batch_size=batch_size, pose=False), verbose=0))
        rgb_test = normalize_probabilities(rgb_model.predict(M1ExactSequence(test, batch_size=batch_size, pose=False), verbose=0))
        pose_dev = normalize_probabilities(pose_model.predict(M1ExactSequence(development, batch_size=batch_size, pose=True), verbose=0))
        pose_test = normalize_probabilities(pose_model.predict(M1ExactSequence(test, batch_size=batch_size, pose=True), verbose=0))
        if level == 'B':
            rf_path = M1_EXPORT / str(run['rf_path'])
            forest = joblib.load(_resolve_model_artifact(rf_path))
            geometry_alpha = float(run['geometry_alpha'])
            same_alpha = float(run['same_rnn_alpha'])
        else:
            forest = RandomForestClassifier(n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=seed + original_seed, n_jobs=-1).fit(geometry_train, labels_for(train))
            rf_path = run_dir / f'geometry_rf_seed_{original_seed}.joblib'
            run_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(forest, rf_path)
            geometry_alpha = _choose_alpha(labels_for(development), rgb_dev, normalize_probabilities(forest.predict_proba(geometry_dev)), geometric=True)
            fused_dev = _geometric_fuse(rgb_dev, normalize_probabilities(forest.predict_proba(geometry_dev)), geometry_alpha)
            same_alpha = _choose_alpha(labels_for(development), pose_dev, fused_dev)
        geometry_probability_dev = normalize_probabilities(forest.predict_proba(geometry_dev))
        geometry_probability_test = normalize_probabilities(forest.predict_proba(geometry_test))
        fused_dev = _geometric_fuse(rgb_dev, geometry_probability_dev, geometry_alpha)
        fused_test = _geometric_fuse(rgb_test, geometry_probability_test, geometry_alpha)
        combined_dev = normalize_probabilities(((1.0 - same_alpha) * pose_dev + same_alpha * fused_dev) * class_weights)
        combined_test = normalize_probabilities(((1.0 - same_alpha) * pose_test + same_alpha * fused_test) * class_weights)
        dev_seed.append(combined_dev)
        test_seed.append(combined_test)
        dev_rgb.append(rgb_dev)
        dev_pose.append(pose_dev)
        dev_geometry.append(geometry_probability_dev)
        test_rgb.append(rgb_test)
        test_pose.append(pose_test)
        test_geometry.append(geometry_probability_test)
        artifacts.extend([rgb_path, pose_path, rf_path])
    return PredictionBundle(probabilities=normalize_probabilities(np.mean(test_seed, axis=0)), development_probabilities=normalize_probabilities(np.mean(dev_seed, axis=0)), branches={'rgb': normalize_probabilities(np.mean(test_rgb, axis=0)), 'pose': normalize_probabilities(np.mean(test_pose, axis=0)), 'geometry': normalize_probabilities(np.mean(test_geometry, axis=0))}, development_branches={'rgb': normalize_probabilities(np.mean(dev_rgb, axis=0)), 'pose': normalize_probabilities(np.mean(dev_pose, axis=0)), 'geometry': normalize_probabilities(np.mean(dev_geometry, axis=0))}, artifacts=artifacts, metadata={'variant': f'M1-32-{level}', 'ensemble_seeds': manifest['seeds']})

def fit_predict_m1(train, development, test, *, level, seed, run_dir, epochs, patience, batch_size) -> PredictionBundle:
    if level in {'B', 'H0'}:
        return _predict_m1_exact(train, development, test, level=level, seed=seed, run_dir=run_dir, batch_size=batch_size)
    base = fit_m0(train, development, level='B', seed=seed, output=run_dir / 'unused.keras', epochs=epochs, patience=patience, batch_size=batch_size)
    base_dev = _predict_keras(base, development, batch_size=batch_size)
    base_test = _predict_keras(base, test, batch_size=batch_size)
    geometry_train = compact_matrix(train, ('geometry',))
    geometry_dev = compact_matrix(development, ('geometry',))
    geometry_test = compact_matrix(test, ('geometry',))
    forest = RandomForestClassifier(n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=-1).fit(geometry_train, labels_for(train))
    forest_path = run_dir / 'geometry_rf.joblib'
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, forest_path)
    rf_dev = forest.predict_proba(geometry_dev)
    rf_test = forest.predict_proba(geometry_test)
    dual_path = run_dir / 'dual.keras'
    dual = fit_m1_dual(train, development, level=level, seed=seed, output=dual_path, epochs=epochs, patience=patience, batch_size=batch_size)
    dual_dev = _predict_keras(dual, development, batch_size=batch_size, dual=True)
    dual_test = _predict_keras(dual, test, batch_size=batch_size, dual=True)
    alpha = _choose_alpha(labels_for(development), dual_dev, rf_dev, geometric=True)
    development_probabilities = _geometric_fuse(dual_dev, rf_dev, alpha)
    probabilities = _geometric_fuse(dual_test, rf_test, alpha)
    return PredictionBundle(probabilities=probabilities, development_probabilities=development_probabilities, branches={'dual': dual_test, 'geometry': rf_test}, development_branches={'dual': dual_dev, 'geometry': rf_dev}, artifacts=[forest_path, dual_path], metadata={'geometry_alpha': alpha, 'variant': f'M1-100-{level}'})
