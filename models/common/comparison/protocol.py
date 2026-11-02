"""Frozen run matrix for the final M0--M4 comparison."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from dataclasses import asdict, dataclass
from itertools import product
MODELS = ('M0', 'M1', 'M2', 'M3', 'M4')
LEVELS = ('H0', 'H1', 'H2')
DATA_REGIMES = ('D0', 'D1', 'D2')
LABELS = ('line', 'other', 'scrum')
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SEEDS = (42,)
CACHE_SIGNATURE = 'a2ccf35ec4d9ab20'
PANEL_V_TRAINABLE_EXCLUSIONS = {'M4': 'decision_2026-08-29: frozen M4 reduced temporal F1 on 6/6 Panel-V folds versus M3; 54 trainable runs cancelled'}

@dataclass(frozen=True)
class Fold:
    panel: str
    fold: str
    train_groups: tuple[str, ...]
    development_group: str
    test_group: str

@dataclass(frozen=True)
class RunSpec:
    panel: str
    model: str
    level: str
    data_regime: str
    fold: str
    seed: int
    train_groups: tuple[str, ...]
    development_group: str
    test_group: str

    @property
    def run_id(self) -> str:
        regime = self.data_regime if self.data_regime else 'NA'
        return f'{self.panel}-{self.model}-{self.level}-{regime}-{self.fold}-S{self.seed}'

    def as_dict(self) -> dict:
        value = asdict(self)
        value['run_id'] = self.run_id
        value['train_groups'] = list(self.train_groups)
        return value

def panel_v_exclusion_reason(spec: RunSpec) -> str:
    """Return a recorded execution-policy exclusion for a Panel-V run."""
    if spec.panel == 'PV' and spec.level != 'B':
        return PANEL_V_TRAINABLE_EXCLUSIONS.get(spec.model, '')
    return ''
PV_FOLDS = (Fold('PV', 'PV01', ('V_3', 'V_4', 'V_5', 'V_6'), 'V_2', 'V_1'), Fold('PV', 'PV02', ('V_1', 'V_4', 'V_5', 'V_6'), 'V_3', 'V_2'), Fold('PV', 'PV03', ('V_1', 'V_2', 'V_5', 'V_6'), 'V_4', 'V_3'), Fold('PV', 'PV04', ('V_1', 'V_2', 'V_3', 'V_6'), 'V_5', 'V_4'), Fold('PV', 'PV05', ('V_1', 'V_2', 'V_3', 'V_4'), 'V_6', 'V_5'), Fold('PV', 'PV06', ('V_2', 'V_3', 'V_4', 'V_5'), 'V_1', 'V_6'))

def historical_folds(group_count: int=5) -> tuple[Fold, ...]:
    """Return the fixed cyclic Panel-H folds over source-disjoint outer groups."""
    if group_count < 3:
        raise ValueError('Panel H requires at least three historical groups')
    groups = tuple((f'G{index:02d}' for index in range(1, group_count + 1)))
    folds = []
    for index, test in enumerate(groups):
        development = groups[(index + 1) % len(groups)]
        train = tuple((group for group in groups if group not in {test, development}))
        folds.append(Fold('PH', f'PH{index + 1:02d}', train, development, test))
    return tuple(folds)

def trainable_runs(*, panels: tuple[str, ...]=('PV', 'PH'), models: tuple[str, ...]=MODELS, levels: tuple[str, ...]=LEVELS, regimes: tuple[str, ...]=DATA_REGIMES, seeds: tuple[int, ...]=SEEDS, historical_group_count: int=5) -> tuple[RunSpec, ...]:
    folds = []
    if 'PV' in panels:
        folds.extend(PV_FOLDS)
    if 'PH' in panels:
        folds.extend(historical_folds(historical_group_count))
    runs = []
    for fold, model, level, regime, seed in product(folds, models, levels, regimes, seeds):
        runs.append(RunSpec(panel=fold.panel, model=model, level=level, data_regime=regime, fold=fold.fold, seed=int(seed), train_groups=fold.train_groups, development_group=fold.development_group, test_group=fold.test_group))
    return tuple(runs)

def frozen_runs(*, historical_group_count: int=5) -> tuple[RunSpec, ...]:
    folds = (*PV_FOLDS, *historical_folds(historical_group_count))
    return tuple((RunSpec(panel=fold.panel, model=model, level='B', data_regime='', fold=fold.fold, seed=0, train_groups=(), development_group=fold.development_group, test_group=fold.test_group) for fold, model in product(folds, MODELS)))

def all_runs(*, historical_group_count: int=5) -> tuple[RunSpec, ...]:
    return frozen_runs(historical_group_count=historical_group_count) + trainable_runs(historical_group_count=historical_group_count)

def select_run(run_id: str, *, historical_group_count: int=5) -> RunSpec:
    matches = [run for run in all_runs(historical_group_count=historical_group_count) if run.run_id == run_id]
    if len(matches) != 1:
        raise KeyError(f'Unknown or ambiguous run_id: {run_id}')
    return matches[0]
