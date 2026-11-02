"""Full batch runtime test on extracted training clips; no checkpoint selection."""
import gc
import time
import numpy as np
from common import *
from model import build,keras,core_hash

def main():
    manifest=load(HERE/'manifest.json');done=np.load(HERE/'features/done.npy')
    indices=[int(i) for i in np.flatnonzero(done) if manifest[i]['split']=='train']
    assert indices
    rows=np.resize(indices,32)
    features=np.load(HERE/'features/multi.npy',mmap_mode='r')
    masks=np.load(HERE/'features/masks.npy',mmap_mode='r')[rows]
    labels=np.array([LABELS.index(manifest[i]['tag']) for i in rows])
    evidence={'passed':False,'batch_size':32,'sequence_length':200,'training_clip_ids':sorted({manifest[i]['clip_id'] for i in rows}),'runs':[]}
    initial=[]
    for mode in MODES:
        keras.backend.clear_session();gc.collect();keras.utils.set_random_seed(17)
        model,core,probe=build(mode);initial.append(core_hash(core))
        x=np.asarray(features[rows,:,0,:] if mode=='fixed' else features[rows])
        start=time.perf_counter();metrics=model.train_on_batch([x,masks],labels,return_dict=True)
        seconds=time.perf_counter()-start
        probabilities=model([x[:2],masks[:2]],training=False).numpy()
        assert probabilities.shape==(2,3) and np.isfinite(probabilities).all()
        evidence['runs'].append({'mode':mode,'train_batch_seconds':seconds,'parameters':model.count_params(),
                                'loss_finite':bool(np.isfinite(metrics['loss']))})
        assert evidence['runs'][-1]['loss_finite']
        del x,model,core,probe;gc.collect()
    assert initial[0]==initial[1]
    evidence['passed']=True;evidence['shared_core_initialization']=True
    save(HERE/'training_smoke.json',evidence);print(evidence,flush=True)

if __name__=='__main__':main()
