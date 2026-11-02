"""Recompute results from predictions, audit caches and verify frozen inputs."""
import pandas as pd
import hashlib
from sklearn.metrics import f1_score,accuracy_score
from common import *

def validate_row_metadata(row,meta,values):
    for key in ['clip_id','video_name','source','source_sha256','active_sha256']:
        if meta[key]!=row[key]:raise RuntimeError('Wrong clip metadata identity: '+row['clip_id'])
    count=min(200,row['frames'])
    expected_indices=np.rint(row['start_s']*row['source_fps']+np.arange(count)*row['source_fps']/25).astype(int).tolist()
    if meta['sample_indices']!=expected_indices:raise RuntimeError('Wrong frame sampling metadata')
    _,geometry=views(np.zeros((row['height'],row['width'],3),np.uint8))
    if meta['views']!=geometry:raise RuntimeError('Wrong crop geometry metadata')
    if values.shape!=(count,9,2048):raise RuntimeError('Wrong feature row shape')
    if hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()!=meta['features_sha256']:
        raise RuntimeError('Feature row checksum mismatch')

def main():
    from train import validate_cache_provenance
    validate_cache_provenance()
    config=check_frozen();parts={s:pd.read_csv(HERE/'partitions'/f'{s}.csv') for s in ['train','validation','test']}
    audit_parts(parts);manifest=load(HERE/'manifest.json')
    selected=load(HERE/'selected_models.json');assert len(selected)==6
    assert np.load(HERE/'features/done.npy').all()
    masks=np.load(HERE/'features/masks.npy',mmap_mode='r');features=np.load(HERE/'features/multi.npy',mmap_mode='r')
    assert features.shape==(len(manifest),200,9,2048)
    for i,row in enumerate(manifest):
        expected=np.arange(200)<min(200,row['frames'])
        np.testing.assert_array_equal(masks[i],expected)
        if not np.isfinite(features[i]).all() or (features[i,~expected]!=0).any():raise RuntimeError('Invalid cache values')
        meta=load(HERE/'clip_metadata'/f'{i:04d}.json')
        validate_row_metadata(row,meta,features[i,expected])
        if digest(row['active_path'])!=row['active_sha256']: raise RuntimeError('Active dataset changed')
    for source,sha in {r['source']:r['source_sha256'] for r in manifest}.items():
        if digest(source)!=sha:raise RuntimeError('Source changed')
    y=np.array([LABELS.index(t) for t in parts['test'].tag]);vy=np.array([LABELS.index(t) for t in parts['validation'].tag])
    index={r['clip_id']:i for i,r in enumerate(manifest)};testmask=masks[[index[c] for c in parts['test'].clip_id]]
    records=[];predictions={m:[] for m in MODES}
    for seed in SEEDS:
        initial=[]
        for mode in MODES:
            directory=HERE/f'seed_{seed}'/mode
            s=next(s for s in selected if s['seed']==seed and s['mode']==mode)
            assert digest(directory/'best.weights.h5')==s['checkpoint_sha256'];initial.append(s['initial_core_sha256'])
            history=pd.read_csv(directory/'history.csv')
            assert int(history.loc[history.val_f1_macro.idxmax(),'epoch'])==s['best_epoch']
            val=np.load(directory/'validation_probabilities.npy')
            assert abs(f1_score(vy,val.argmax(1),average='macro')-s['val_f1_macro'])<1e-12
            frame=pd.read_csv(directory/'test_predictions.csv');probs=np.load(directory/'test_probabilities.npy')
            assert frame.clip_id.equals(parts['test'].clip_id) and frame.video_name.equals(parts['test'].video_name) and frame.tag.equals(parts['test'].tag)
            assert probs.shape==(len(y),3) and np.isfinite(probs).all()
            np.testing.assert_allclose(probs.sum(1),1,atol=1e-5)
            np.testing.assert_allclose(probs,frame[['p_'+k for k in LABELS]],atol=1e-7)
            pred=probs.argmax(1);assert frame.prediction.tolist()==[LABELS[i] for i in pred]
            metrics=score(y,pred);saved=load(directory/'test_metrics.json')
            assert saved==metrics
            assert abs(f1_score(y,pred,average='macro')-metrics['f1_macro'])<1e-12
            assert abs(accuracy_score(y,pred)-metrics['accuracy'])<1e-12
            if mode=='multi':
                attn=np.load(directory/'test_attention.npy')
                assert attn.shape==(len(y),200,9) and np.isfinite(attn).all() and (attn>=0).all()
                np.testing.assert_allclose(attn[testmask].sum(-1),1,atol=1e-5)
                assert (attn[~testmask]==0).all()
                save(directory/'attention_summary.json',{'mean_weights':attn[testmask].mean(0).tolist(),
                     'view0':'original repository center','interpretation':'Learned weights, not causal explanations'})
            records.append({'seed':seed,'mode':mode,'best_epoch':s['best_epoch'],
                            **{k:v for k,v in metrics.items() if k!='confusion_matrix'}})
            predictions[mode].append(pred)
        assert initial[0]==initial[1]
    rows=pd.DataFrame(records);rows.to_csv(HERE/'results.csv',index=False)
    interval=paired_bootstrap(y,predictions['fixed'],predictions['multi'],replicates=2000,seed=20261024)
    save(HERE/'paired_bootstrap.json',interval)
    means=rows.groupby('mode').mean(numeric_only=True);lo,hi=interval['ci95']
    conclusion='Mejora' if lo>0 else ('Empeora' if hi<0 else 'Diferencia no concluyente al 95 %')
    lines=['# Central frente a central + ocho regiones con atención','',conclusion,'',
           'Promedios de tres semillas; porcentajes y diferencias en puntos porcentuales.','',
           '| Métrica | Central | Nueve vistas | Diferencia |','|---|---:|---:|---:|']
    for key in ['f1_macro','accuracy','line_f1','line_recall','scrum_f1','scrum_recall']:
        a,b=means.loc['fixed',key],means.loc['multi',key]
        lines.append(f'| {key} | {100*a:.2f} | {100*b:.2f} | {100*(b-a):+.2f} |')
    lines += ['',f"F1 macro: diferencia {100*interval['mean_difference']:+.2f} pp; IC95 [{100*lo:+.2f}, {100*hi:+.2f}] pp.",'',
      '2890 clips corregidos seleccionados por las mismas identidades históricas; 204 fuentes documentadas. '
      '1260 filas de entrenamiento (420/clase), 980 validación y 817 test; hasta200 cuadros a25FPS. '
      'ResNet50 FP32 congelada; BiGRU64/32; seeds17,42,2026; selección por validación antes de test.','',
      'El control y la vista0 del modelo nuevo son exactamente la misma ventana central, calculada con factor0.5 '
      'como en el repositorio corregido. Las ocho vistas de contexto usan altura360 y rejilla2x4. '
      'Ambos consumen los mismos cuadros originales sin recompresión. La atención añade parámetros y contexto a varias escalas; '
      'esta comparación no separa la contribución de esos factores.','',
      'Evaluación histórica por clips; no es prueba externa independiente. Puede haber dependencia por partido, '
      'incluidos partidos de origen compartidos entre entrenamiento y validación. '
      'El bootstrap por clips puede subestimar incertidumbre. Los pesos de atención no prueban causalidad.','',
      'Predicciones, checkpoints, historias y atención de test se conservan en seed_*/.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(HERE/'verification.json',{'passed':True,'clips':len(manifest),'models':6,'metrics_verified_with_sklearn':True,
         'active_and_source_hashes_intact':True,'center_view0_shared':True,'padding_masks_attention_verified':True})
    save(HERE/'completion.json',{'complete':True,'conclusion':conclusion,**interval})
    save(HERE/'status.json',{'phase':'complete','conclusion':conclusion,**interval})
    print('\n'.join(lines[:18]),flush=True)

if __name__=='__main__':main()
