"""Resolve historical model names to one local copy of each trained artifact."""
from functools import lru_cache
import fnmatch
import csv
import json
import os
from pathlib import Path
import tempfile
from models.source_layout import source_path, legacy_name

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT.parent/('archivo_'+ROOT.name+'_20261031')/'limpieza_informes_20261031'
_TEMPORARY = []


@lru_cache(maxsize=1)
def index():
    return json.loads((ROOT/'models/familias/catalogo.json').read_text(encoding='utf-8'))


def _relative(path):
    text = os.fspath(path).replace('\\', '/')
    for base in [ROOT.as_posix(), index()['source_root'].replace('\\', '/')]:
        if text.casefold() == base.casefold():
            return ''
        if text.casefold().startswith(base.casefold().rstrip('/')+'/'):
            return text[len(base.rstrip('/'))+1:]
    return None if Path(text).is_absolute() or (len(text)>1 and text[1]==':') else text


def resolve_artifact(path):
    """Unknown paths in the original never fall back to its files."""
    if not isinstance(path, (str, os.PathLike)):
        return path
    absolute=Path(path)
    if absolute.is_absolute() and absolute.is_relative_to(ARCHIVE_ROOT):
        target=absolute.resolve()
        if not target.is_relative_to(ARCHIVE_ROOT.resolve()):
            raise ValueError('El artefacto debe permanecer dentro del archivo')
        return target
    relative = _relative(path)
    if relative is None:
        return Path(path)
    aliases = index()['aliases']
    key = relative.replace('models/M0/', 'models/inceptionv3_gru/')
    text = os.fspath(path).replace('\\', '/')
    explicit_current = text.casefold().startswith(ROOT.as_posix().casefold().rstrip('/')+'/')
    historical = index().get('historical_media_aliases',{})
    target = historical.get(relative) if not explicit_current else None
    if target is None:
        target = aliases.get(relative, aliases.get(legacy_name(relative), aliases.get(key)))
    archived = index().get('archived_reports', {}).get(target if target is not None else relative)
    if archived is not None:
        report = (ARCHIVE_ROOT/archived).resolve()
        if not report.is_relative_to(ARCHIVE_ROOT.resolve()):
            raise ValueError('El informe debe permanecer dentro del archivo')
        return report
    candidate = source_path(ROOT/(target if target is not None else relative)).resolve()
    legacy_root = ROOT / 'experiments'
    if target is None and candidate.is_relative_to(legacy_root):
        candidate = ROOT / 'data' / candidate.relative_to(legacy_root)
    if not candidate.is_relative_to(ROOT.resolve()):
        raise ValueError('El artefacto debe permanecer dentro de la entrega')
    original = index()['source_root'].replace('\\', '/').rstrip('/')
    from_original = text.casefold().startswith(original.casefold()+'/') and not text.casefold().startswith(ROOT.as_posix().casefold()+'/')
    if from_original and target is None and not candidate.exists():
        raise FileNotFoundError(f'Artefacto excluido o no incorporado: {relative}')
    return candidate


def artifact_glob(directory, pattern, recursive=False):
    directory = Path(directory)
    actual = resolve_artifact(directory)
    found = {str(p): p for p in (actual.rglob(pattern) if recursive else actual.glob(pattern))}
    relative = _relative(directory)
    patterns = tuple(str(pattern).replace('\\', '/').split('/'))
    if recursive:
        patterns = ('**',)+patterns
    def matches(parts, rules):
        if not rules:
            return not parts
        if rules[0]=='**':
            return matches(parts,rules[1:]) or bool(parts) and matches(parts[1:],rules)
        return bool(parts) and fnmatch.fnmatch(parts[0],rules[0]) and matches(parts[1:],rules[1:])
    if relative is not None:
        prefixes = tuple(dict.fromkeys((relative.rstrip('/')+'/', legacy_name(relative).rstrip('/')+'/')))
        for alias in index()['aliases']:
            prefix = next((value for value in prefixes if alias.startswith(value)), None)
            if prefix is not None:
                tail = alias[len(prefix):]
                parts=tuple(tail.split('/'))
                for length in range(1,len(parts)+1):
                    if matches(parts[:length],patterns):
                        logical = ROOT/relative/Path(*parts[:length])
                        found[str(logical)] = logical
    return iter(found[key] for key in sorted(found))


def temporary_settings():
    """YOLO settings are transient and do not create a repository cache."""
    folder = tempfile.TemporaryDirectory(prefix='tesina-model-settings-')
    _TEMPORARY.append(folder)
    return Path(folder.name)


def load_array(path,*args,**kwargs):
    """Read small archived result arrays from CSV; never recreate feature banks."""
    import numpy as np
    relative=_relative(path) if isinstance(path,(str,os.PathLike)) else None
    arrays=index().get('result_arrays',{})
    record=arrays.get(relative,arrays.get(legacy_name(relative),arrays.get(relative.replace('models/M0/','models/inceptionv3_gru/')))) if relative is not None else None
    if record is None:
        return np.load(resolve_artifact(path),*args,**kwargs)
    target=(ROOT/record['path']).resolve()
    if not target.is_relative_to(ROOT.resolve()):raise ValueError('Resultado fuera de la entrega')
    target=resolve_artifact(target)
    with target.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.reader(handle))[1:]
    return np.asarray([row[1:] for row in rows],dtype=record['dtype']).reshape(record['shape'])
