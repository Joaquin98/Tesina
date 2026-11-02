"""Public M0 loader; trained artifacts are resolved by the shared catalog."""
from pathlib import Path


def load_model(project_root=None):
    root=Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    from .web_adapter import RugbyPlayDetector
    model = RugbyPlayDetector(str(root/'models/M0'),model_name='best_model/best_model.keras',extractor_name='resnet50')
    from models.common.features import FeatureProvider
    model.feature_provider = FeatureProvider('M0')
    return model
