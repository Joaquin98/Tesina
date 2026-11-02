import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings

class M1H2D2CrossFoldDetector(PlayerGeometryFusionDetector):
    """Web ensemble built from the six completed M1-H2-D2 outer folds.

    ResNet-50 and YOLO are evaluated once per sampled frame.  Each temporal
    window is then passed through the six fold-specific dual BiGRU models and
    geometry forests.  Their already-fused probabilities receive equal-weight
    soft voting.  The web ensemble itself is intentionally marked exploratory:
    the reported metrics belong to the outer-fold protocol, not to this new
    cross-fold voting policy.
    """
    RGB_LENGTH = 200
    GEOMETRY_FRAMES = 200
    POSE_FRAMES_PER_WINDOW = 100
    EVENT_NEIGHBOR_BOOST = 0.0

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.export_root = self.project_root / M1_H2_D2_WEB_EXPORT_RELATIVE
        self.manifest_path = SELECTED_MANIFEST_PATH
        if not _resolve_model_artifact(self.manifest_path).exists():
            raise FileNotFoundError(f'M1-H2-D2 web manifest not found: {self.manifest_path}')
        self.manifest = json.loads(_resolve_model_artifact(self.manifest_path).read_text(encoding='utf-8'))
        if self.manifest.get('model_name') != M1_H2_D2_WEB_MODEL_NAME:
            raise ValueError('Unexpected M1-H2-D2 web manifest model name')
        if self.manifest.get('label_order') != ['line', 'other', 'scrum']:
            raise ValueError('Unexpected M1-H2-D2 label order')
        if len(self.manifest.get('folds', [])) != EXPECTED_MEMBERS:
            raise ValueError('M1-H2-D2 web ensemble requires exactly six folds')
        self.pose_model_path = self.project_root / self.manifest.get('pose_model', PLAYER_GEOMETRY_FUSION_POSE_MODEL)
        if not _resolve_model_artifact(self.pose_model_path).exists():
            raise FileNotFoundError(f'Pose checkpoint not found: {self.pose_model_path}')
        model_dir = self.project_root / 'models' / 'M0'
        RugbyPlayDetector.__init__(self, str(model_dir), model_name='best_model/best_model.keras', extractor_name='resnet50')
        saved_config = sys.modules.pop('config', None)
        sys.path.insert(0, str(source_directory(player_geometry_path)))
        try:
            from models.M1.pose_cache import detect_frame
            from models.M1.geometry_features import build_geometry_features
        finally:
            sys.path.remove(str(source_directory(player_geometry_path)))
            if saved_config is not None:
                sys.modules['config'] = saved_config
        self._detect_frame = detect_frame
        self._build_geometry_features = build_geometry_features
        self.fold_models = []
        for fold in self.manifest['folds']:
            dual_path = self.project_root / fold['dual_model']
            forest_path = self.project_root / fold['geometry_rf']
            missing = [path for path in (dual_path, forest_path) if not _resolve_model_artifact(path).exists()]
            if missing:
                raise FileNotFoundError(f'Missing M1-H2-D2 fold artifacts: {missing}')
            dual_model = keras.models.load_model(_resolve_model_artifact(dual_path), compile=False)
            forest = joblib.load(_resolve_model_artifact(forest_path))
            if int(getattr(forest, 'n_features_in_', -1)) != 549:
                raise ValueError(f'Unexpected geometry input for {forest_path}')
            self.fold_models.append({'fold': fold['fold'], 'dual': dual_model, 'geometry': forest, 'geometry_alpha': float(fold['geometry_alpha']), 'line_threshold': float(fold['line_threshold']), 'scrum_threshold': float(fold['scrum_threshold'])})
        try:
            yolo_config_dir = _temporary_model_settings()
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault('YOLO_CONFIG_DIR', str(yolo_config_dir))
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('M1-H2-D2 web inference requires ultralytics') from exc
        self.pose_detector = YOLO(_resolve_model_artifact(str(self.pose_model_path)))
        self.label_vocab = np.asarray(self.manifest['label_order'], dtype=object)
        self._geometry_frames = []
        self._geometry_positions = np.zeros((0, 31), dtype=np.float32)
        self._geometry_poses = np.zeros((0, 30), dtype=np.float32)
        self._geometry_mask = np.zeros((0,), dtype=bool)
        self.fusion_metadata = {'deployment_policy': self.manifest['deployment_policy'], 'fold_count': len(self.fold_models), 'validation_status': self.manifest['validation_status'], 'recommended_web_thresholds': self.manifest['recommended_web_thresholds'], 'cross_validation_metrics': self.manifest['cross_validation_metrics']}

    @property
    def display_name(self) -> str:
        return 'M1-H2-D2 Â· ensemble de 6 folds'

    @staticmethod
    def _ordered_forest_probability(forest, geometry_vector: np.ndarray) -> np.ndarray:
        raw = np.asarray(forest.predict_proba(geometry_vector)[0], dtype=np.float64)
        ordered = np.zeros(3, dtype=np.float64)
        for index, label_index in enumerate(np.asarray(forest.classes_).astype(int)):
            ordered[int(label_index)] = raw[index]
        return ordered

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        positions = self._geometry_positions[start_frame:end_frame]
        poses = self._geometry_poses[start_frame:end_frame]
        pose_mask = self._geometry_mask[start_frame:end_frame].astype(bool)
        if len(positions) > self.POSE_FRAMES_PER_WINDOW:
            selected = np.linspace(0, len(positions) - 1, self.POSE_FRAMES_PER_WINDOW, dtype=np.int32)
            positions = positions[selected]
            poses = poses[selected]
            pose_mask = pose_mask[selected]
        pose_values = np.concatenate([positions, poses], axis=1)
        aligned_pose = self._align(pose_values, self.RGB_LENGTH)
        aligned_pose_mask = self._align(pose_mask.astype(np.float32)[:, None], self.RGB_LENGTH)[:, 0]
        pose_input = np.concatenate([aligned_pose, aligned_pose_mask[:, None]], axis=1)
        rgb_input = np.asarray(padded_features, dtype=np.float32)
        rgb_mask = np.asarray(mask, dtype=bool)
        valid_geometry = pose_values[pose_mask] if np.any(pose_mask) else pose_values
        if len(valid_geometry) == 0:
            valid_geometry = np.zeros((1, 61), dtype=np.float32)
        deltas = np.diff(valid_geometry, axis=0) if len(valid_geometry) > 1 else np.zeros((1, 61), dtype=np.float32)
        geometry_vector = np.concatenate([valid_geometry.mean(axis=0), valid_geometry.std(axis=0), valid_geometry.min(axis=0), valid_geometry.max(axis=0), np.quantile(valid_geometry, 0.25, axis=0), np.quantile(valid_geometry, 0.75, axis=0), deltas.mean(axis=0), deltas.std(axis=0), np.abs(deltas).mean(axis=0)]).astype(np.float32).reshape(1, -1)
        fold_probabilities = []
        for item in self.fold_models:
            dual_probability = np.asarray(item['dual']({'rgb': rgb_input[None, ...], 'pose': pose_input[None, ...], 'sequence_mask': rgb_mask[None, ...]}, training=False)[0].numpy(), dtype=np.float64)
            geometry_probability = self._ordered_forest_probability(item['geometry'], geometry_vector)
            alpha = float(item['geometry_alpha'])
            fused = np.exp((1.0 - alpha) * np.log(np.clip(dual_probability, 1e-07, 1.0)) + alpha * np.log(np.clip(geometry_probability, 1e-07, 1.0)))
            fold_probabilities.append(self._normalize(fused))
        averaged = self._normalize(np.mean(fold_probabilities, axis=0))
        return averaged[None, :].astype(np.float32)
