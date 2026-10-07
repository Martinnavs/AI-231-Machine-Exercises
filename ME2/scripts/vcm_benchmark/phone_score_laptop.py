import numpy as np,soundfile as sf,json,shutil
from pathlib import Path
from types import SimpleNamespace
from calib import fit,apply
import replay_bench as R
sr=16000
al=sf.read('phone/aligned_laptop.wav')[0].astype('float32'); cl=sf.read('runs/full/stream.wav')[0].astype('float32')
res={r['order']:r for r in json.load(open('phone/align_laptop.json'))}
plan=json.load(open('runs/full/plan.json')); keep=[t for t in plan if res[t['order']]['ok']]
T=max(t['window_end'] for t in keep)
al=al[:int((T+4)*sr)]
g=fit(cl[:int(300*sr)],al[:int(300*sr)])[0]; np.save('phone/gain_map_laptop.npy',g)
mapped=apply(al,g)
for name,x in (('laptop_raw',al),('laptop_mapped',mapped)):
    run=Path('runs')/name; run.mkdir(exist_ok=True)
    sf.write(run/'stream.wav',x,sr,subtype='PCM_16'); json.dump(keep,open(run/'plan.json','w')); shutil.copy('runs/full/config.json',run/'config.json')
    R.stream(SimpleNamespace(run=str(run))); R.do_score(SimpleNamespace(run=str(run)))
print('DONE',len(keep))
