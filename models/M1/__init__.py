"""Public M1 loader; trained artifacts are resolved by the shared catalog."""
from pathlib import Path


def load_model(project_root=None):
    root=Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    from .inference import M1H2D2CrossFoldDetector
    model = M1H2D2CrossFoldDetector(root)
    from models.common.features import FeatureProvider
    model.feature_provider = FeatureProvider('M1')
    return model
