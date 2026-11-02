"""Six paired runs with validation selection before any test prediction."""
import gc
import time
import pandas as pd
from common import *
from model import build,core_hash,keras,tf

def validate_cache_provenance(directory=HERE):
    provenance=load(directory/'features/provenance.json')
    if provenance['protocol']!=digest(directory/'protocol.json'):raise RuntimeError('Cache protocol changed')
    for name,expected in provenance['code'].items():
        if digest(directory/name)!=expected:raise RuntimeError('Feature extraction code changed')
    complete=directory/'extraction_complete.json'
    if complete.exists() and load(complete)['protocol_sha256']!=provenance['protocol']:
        raise RuntimeError('Completion marker belongs to another protocol')

class Store:
    def __init__(self):
        validate_cache_provenance()
        self.rows=load(HERE/'manifest.json')
        self.index={r['clip_id']:i for i,r in enumerate(self.rows)}
        self.features=np.load(HERE/'features/multi.npy',mmap_mode='r')
        self.mask=np.load(HERE/'features/masks.npy',mmap_mode='r')
        if not np.load(HERE/'features/done.npy').all(): raise RuntimeError('Incomplete extraction')
    def indices(self,df): return np.array([self.index[c] for c in df.clip_id])
    def batch(self,rows,mode):
        x=self.features[rows,:,0,:] if mode=='fixed' else self.features[rows]
        return np.asarray(x),np.asarray(self.mask[rows])

def labels(df): return np.array([LABELS.index(t) for t in df.tag],np.int32)

def predictor(model,mode):
    shape=[None,200,2048] if mode=='fixed' else [None,200,9,2048]
    @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32),tf.TensorSpec([None,200],tf.bool)])
    def infer(x,mask): return model([x,mask],training=False)
    return infer

def predict(infer,store,df,mode,batch_size=32):
    rows=store.indices(df);result=[]
    for start in range(0,len(rows),batch_size): result.append(infer(*store.batch(rows[start:start+batch_size],mode)).numpy())
    result=np.concatenate(result)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1),1,atol=1e-5): raise RuntimeError('Invalid probabilities')
    return result

def fit(store,parts,settings,seed,mode):
    directory=HERE/f'seed_{seed}'/mode;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'selected.json').exists():
        selected=load(directory/'selected.json')
        if digest(directory/'best.weights.h5')!=selected['checkpoint_sha256']: raise RuntimeError('Checkpoint changed')
        return selected
    keras.backend.clear_session();gc.collect();keras.utils.set_random_seed(seed)
    model,core,_=build(mode);initial=core_hash(core)
    initial_path=HERE/f'seed_{seed}'/'initialization.json'
    if initial_path.exists():
        if load(initial_path)['core_sha256']!=initial: raise RuntimeError('Shared initialization mismatch')
    else: save(initial_path,{'seed':seed,'core_sha256':initial})
    infer=predictor(model,mode);rows=store.indices(parts['train']);train_y=labels(parts['train']);val_y=labels(parts['validation'])
    rng=np.random.default_rng(seed);best=-1.;waiting=0;history=[];start=time.perf_counter()
    for epoch in range(1,settings['max_epochs']+1):
        es=time.perf_counter();order=rng.permutation(len(rows));model.reset_metrics()
        for offset in range(0,len(order),settings['batch_size']):
            pos=order[offset:offset+settings['batch_size']]
            x,mask=store.batch(rows[pos],mode)
            metrics=model.train_on_batch([x,mask],train_y[pos],return_dict=True)
        probabilities=predict(infer,store,parts['validation'],mode,settings['batch_size'])
        val=score(val_y,probabilities.argmax(1))
        history.append({'epoch':epoch,'train_loss':float(metrics['loss']),'train_accuracy':float(metrics['accuracy']),
                        'val_f1_macro':val['f1_macro'],'val_accuracy':val['accuracy'],'seconds':time.perf_counter()-es})
        if val['f1_macro']>best:
            best=val['f1_macro'];best_epoch=epoch;waiting=0
            model.save_weights(directory/'best.weights.h5');np.save(directory/'validation_probabilities.npy',probabilities)
        else: waiting+=1
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        status={'phase':'training','seed':seed,'mode':mode,**history[-1],'best_epoch':best_epoch}
        save(HERE/'status.json',status);print(json.dumps(status),flush=True)
        if waiting>=settings['patience']:break
    selected={'seed':seed,'mode':mode,'best_epoch':best_epoch,'epochs':len(history),'val_f1_macro':best,
              'initial_core_sha256':initial,'checkpoint_sha256':digest(directory/'best.weights.h5'),
              'seconds':time.perf_counter()-start,'parameters':model.count_params()}
    save(directory/'selected.json',selected);return selected

def main():
    config=check_frozen();settings=config['training']
    parts={s:pd.read_csv(HERE/'partitions'/f'{s}.csv') for s in ['train','validation','test']};audit_parts(parts)
    assert parts['train'].tag.value_counts().to_dict()=={'line':420,'other':420,'scrum':420}
    store=Store();selections=[]
    for seed in SEEDS:
        for mode in MODES: selections.append(fit(store,parts,settings,seed,mode))
    path=HERE/'selected_models.json'
    if path.exists():
        if load(path)!=selections:raise RuntimeError('Frozen selections changed')
    else:save(path,selections)
    for selected in selections:
        seed,mode=selected['seed'],selected['mode'];directory=HERE/f'seed_{seed}'/mode
        if (directory/'test_metrics.json').exists():continue
        keras.backend.clear_session();gc.collect();keras.utils.set_random_seed(seed)
        model,_,probe=build(mode);model.load_weights(directory/'best.weights.h5')
        probabilities=predict(predictor(model,mode),store,parts['test'],mode)
        np.save(directory/'test_probabilities.npy',probabilities)
        output=parts['test'].copy();output['prediction']=[LABELS[i] for i in probabilities.argmax(1)]
        for i,label in enumerate(LABELS):output['p_'+label]=probabilities[:,i]
        output.to_csv(directory/'test_predictions.csv',index=False)
        if probe is not None:
            rows=store.indices(parts['test']);weights=[]
            for start in range(0,len(rows),8):
                x,mask=store.batch(rows[start:start+8],mode);weights.append(probe([x,mask],training=False).numpy())
            np.save(directory/'test_attention.npy',np.concatenate(weights))
        save(directory/'test_metrics.json',score(labels(parts['test']),probabilities.argmax(1)))
    save(HERE/'training_complete.json',{'models':6,'test_clips':len(parts['test']),'selections_sha256':digest(path)})

if __name__=='__main__':
    with writer_lock('training.lock'):main()
