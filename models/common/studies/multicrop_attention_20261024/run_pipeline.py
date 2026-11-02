"""Single-owner resumable background experiment; failures stay visible."""
import os
import sys
import subprocess
import time
import msvcrt
from common import *

def main():
    lock=(HERE/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError as e:raise RuntimeError('Another instance owns the experiment') from e
    hashes={p.name:digest(p) for p in HERE.glob('*.py')}
    hashes.update({name:digest(HERE/name) for name in ['protocol.json','manifest.json']})
    frozen=HERE/'implementation_hashes.json'
    if frozen.exists():
        if load(frozen)!=hashes:raise RuntimeError('Frozen implementation changed')
    else:save(frozen,hashes)
    check_frozen()
    save(HERE/'run.json',{'pid':os.getpid(),'python':sys.executable,'started_unix':time.time(),'cwd':str(ROOT)})
    with (HERE/'run.log').open('a',encoding='utf8',buffering=1) as log:
        for script in ['extract.py','train.py','report.py']:
            child=subprocess.Popen([sys.executable,'-u',str(HERE/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(HERE/'process.json',{'pipeline_pid':os.getpid(),'child_pid':child.pid,'script':script,'started_unix':time.time()})
            code=child.wait()
            if code:
                save(HERE/'status.json',{'phase':'failed','script':script,'exit_code':code})
                raise RuntimeError(f'{script} failed: {code}; see run.log')

if __name__=='__main__':main()
