"""Prepare cache, train, evaluate and predict with local M0--M3 models."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from models.model_catalog import index, resolve_artifact, temporary_settings


def detector(family):
    """Frozen historical deployments; model choice does not retrain any weights."""
    from models.common.workflow import configure, load_detector
    configure(2)
    return load_detector(family)



def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['list','verify','check','cache','train','predict','evaluate'])
    parser.add_argument('--family',choices=['M0','M1','M2','M3'])
    parser.add_argument('--video',type=Path)
    parser.add_argument('--no-cache',action='store_true',help='Recalcular las características sin leer ni guardar caché')
    parser.add_argument('--cache-dir',type=Path,help='Carpeta local de caché (predeterminada: .runtime/cache/features)')
    parser.add_argument('--run',type=Path,help='Corrida guardada para inferir, evaluar o continuar entrenamiento con un optimizador nuevo')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/clips_dataset')
    parser.add_argument('--split',choices=['all','train','validation','test'],default=None)
    parser.add_argument('--output',type=Path,help='Nueva carpeta de corrida; nunca se sobrescribe')
    parser.add_argument('--epochs',type=int,default=40,help='Épocas de las cabezas neuronales; M2 ajusta regresión logística')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--init',choices=['published','random'],default='published',help='Inicialización de las cabezas neuronales; los extractores quedan congelados')
    parser.add_argument('--limit-per-class',type=int,help='Limitar ejemplos por clase y partición para pruebas pequeñas')
    args=parser.parse_args(argv)
    if args.command not in ['list','verify'] and not args.family:parser.error('--family es obligatorio')
    if args.command=='predict' and not args.video:parser.error('--video es obligatorio')
    if args.command=='train' and (args.video is not None or args.split is not None):parser.error('train utiliza --dataset con train.csv y validation.csv; no --video ni --split')
    if args.command=='cache' and args.run is not None:parser.error('--run se utiliza para train, predict, evaluate o check')
    if args.command=='list':
        records=[r for r in index()['runs'] if args.family is None or r['model']==args.family]
        print(json.dumps({'families':index()['families'],'runs':records,'unique_weight_files':len(index()['objects'])},ensure_ascii=False))
    elif args.command=='verify':
        catalog=index()
        for expected,item in catalog['objects'].items():
            path=ROOT/item['path']
            with path.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==expected,path
        checked_aliases=0
        archived_aliases=0
        for alias,stored in catalog['aliases'].items():
            if stored in catalog.get('archived_reports',{}):
                archived_aliases+=1
                continue
            target=resolve_artifact(ROOT/alias)
            assert target.exists(),alias
            if alias.endswith('.weights.h5'):
                assert target.name.endswith('.weights.h5'), 'El sufijo selecciona el formato del loader Keras: '+alias
            checked_aliases+=1
        print(json.dumps({'status':'ok','unique_weight_files':len(catalog['objects']),'aliases':checked_aliases,'archived_result_aliases':archived_aliases}))
    else:
        with contextlib.redirect_stdout(sys.stderr):
            from models.common import workflow
            workflow.configure(args.threads)
            if args.command=='cache':
                if args.no_cache:parser.error('cache requiere guardar la caché')
                result=workflow.prepare_cache(args.family,dataset=args.dataset,split=args.split or 'all',video=args.video,cache_dir=args.cache_dir,limit_per_class=args.limit_per_class)
            elif args.command=='train':
                if args.no_cache:parser.error('train utiliza características guardadas en la caché local')
                result=workflow.train_run(args.family,dataset=args.dataset,output=args.output,cache_dir=args.cache_dir,epochs=args.epochs,batch_size=args.batch_size,patience=args.patience,seed=args.seed,threads=args.threads,init=args.init,initial_run=args.run,limit_per_class=args.limit_per_class)
            elif args.command=='predict':
                result=workflow.predict(args.family,args.video,run=args.run,use_cache=not args.no_cache,cache_dir=args.cache_dir)
            elif args.command=='evaluate':
                if args.split=='all':parser.error('evaluate requiere una partición concreta')
                result=workflow.evaluate(args.family,dataset=args.dataset,split=args.split or 'test',run=args.run,cache_dir=args.cache_dir,use_cache=not args.no_cache,limit_per_class=args.limit_per_class)
            else:
                model=workflow.load_detector(args.family,args.run)
                result={'status':'ok','family':args.family,'model_loaded':True,'feature_cache_enabled':not args.no_cache}
        print(json.dumps(result,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))


if __name__=='__main__':main()
