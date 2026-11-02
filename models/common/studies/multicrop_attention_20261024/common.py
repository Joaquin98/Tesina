"""Frozen experiment helpers. View zero is ALWAYS the existing central crop."""
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(parent for parent in _LayoutPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from models.source_layout import experiment_directory
from pathlib import Path
import importlib.util
import json
import cv2
import numpy as np

HERE=experiment_directory(__file__)
REPO=HERE.parents[1]
ROOT=HERE.parents[3]
PREVIOUS=ROOT/'data/yolo_crop_classification_20261023'
spec=importlib.util.spec_from_file_location('legacy_protocol',PREVIOUS/'common.py')
legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
save,digest,score,audit_parts,paired_bootstrap=legacy.save,legacy.digest,legacy.score,legacy.audit_parts,legacy.paired_bootstrap
LABELS=['line','other','scrum']; SEEDS=[17,42,2026]; MODES=['fixed','multi']

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def context_geometry(width,height):
    scale=max(360/height,224/width)
    nw,nh=max(224,int(np.floor(width*scale+.5))),max(224,int(np.floor(height*scale+.5)))
    xs=np.rint(np.linspace(0,nw-224,4)).astype(int)
    ys=np.rint(np.linspace(0,nh-224,2)).astype(int)
    rects=[[int(x),int(y),224,224] for y in ys for x in xs]
    if len(set(map(tuple,rects)))!=8 or max(np.diff(xs))>224 or max(np.diff(ys))>224:
        raise ValueError('Unsupported aspect ratio for eight covering context crops')
    return nw,nh,rects

def views(frame):
    h,w=frame.shape[:2]
    hw,hh=round(w*.5),round(h*.5)
    if min(hw,hh)<224: raise ValueError('Original source too small for repository crop')
    half=cv2.resize(frame,(hw,hh),interpolation=cv2.INTER_LINEAR)
    x,y=(hw-224)//2,(hh-224)//2
    images=[half[y:y+224,x:x+224].copy()]
    meta=[{'view':0,'kind':'repository_center','resize':[hw,hh],'rect':[x,y,224,224]}]
    nw,nh,rects=context_geometry(w,h)
    normalized=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_LINEAR)
    for i,(x,y,ww,hx) in enumerate(rects,1):
        images.append(normalized[y:y+hx,x:x+ww].copy())
        meta.append({'view':i,'kind':'context','resize':[nw,nh],'rect':[x,y,ww,hx]})
    return np.stack(images),meta

def read_samples(row):
    """Same source indices used to export the corrected 25 FPS clips."""
    cap=cv2.VideoCapture(row['source'])
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot open '+row['source'])
        fps=cap.get(cv2.CAP_PROP_FPS)
        indices=np.rint(row['start_s']*fps+np.arange(min(200,row['frames']))*fps/25).astype(int)
        if indices[-1]>=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)): raise RuntimeError('Window exceeds source')
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(indices[0]))
        if abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-indices[0])>.1: raise RuntimeError('Seek failed')
        decoded=int(indices[0])-1
        for index in indices:
            while decoded<index:
                ok,frame=cap.read()
                if not ok: raise RuntimeError('Incomplete source')
                decoded+=1
            yield int(index),frame
    finally: cap.release()

def check_frozen():
    frozen=HERE/'implementation_hashes.json'
    if frozen.exists():
        for name,expected in load(frozen).items():
            if digest(HERE/name)!=expected: raise RuntimeError('Frozen implementation changed: '+name)
    p=load(HERE/'protocol.json')
    for path,expected in p['input_hashes'].items():
        if digest(path)!=expected: raise RuntimeError('Frozen input changed: '+path)
    for name,expected in p['partition_hashes'].items():
        if digest(HERE/'partitions'/name)!=expected: raise RuntimeError('Partition changed')
    if digest(HERE/'manifest.json')!=p['manifest_sha256']: raise RuntimeError('Manifest changed')
    return p

def writer_lock(name):
    import msvcrt
    lock=(HERE/name).open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        lock.close();raise RuntimeError('Another writer owns '+name)
    return lock
