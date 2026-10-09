import numpy as np,soundfile as sf,json,shutil
from pathlib import Path
from types import SimpleNamespace
from agc import agc
import replay_bench as R
sr=16000
al=sf.read('phone/aligned_laptop.wav')[0].astype('float32'); cl=sf.read('runs/full/stream.wav')[0].astype('float32')
plan=json.load(open('runs/laptop_raw/plan.json')); T=max(t['window_end'] for t in plan); al=al[:int((T+4)*sr)]
def lv(x,fr=320):
    n=len(x)//fr*fr;e=10*np.log10((x[:n].reshape(-1,fr)**2).mean(1)+1e-12);return e
ec=lv(cl[:300*sr]);er=lv(al[:300*sr])
loud=lambda e:10*np.log10(np.mean(10**(np.sort(e)[-len(e)//10:]/10)))
g=loud(ec)-loud(er);print('clean loudest-10% level',round(loud(ec),1),'recorded',round(loud(er),1),'-> fixed gain',round(g,1),'dB')
variants={'laptop_gain':np.clip(al*10**(g/20),-1,1).astype('float32'),'laptop_agc':agc(al,target_dbfs=loud(ec))}
for name,x in variants.items():
    run=Path('runs')/name;run.mkdir(exist_ok=True)
    sf.write(run/'stream.wav',x,sr,subtype='PCM_16');json.dump(plan,open(run/'plan.json','w'));shutil.copy('runs/full/config.json',run/'config.json')
    R.stream(SimpleNamespace(run=str(run)));R.do_score(SimpleNamespace(run=str(run)))
print('DONE')
