"""Align every trial of the recording to the clean timeline individually; build a clean-timeline stream from the recorded segments."""
import numpy as np,soundfile as sf,json
sr=16000;OFF0=10.35;f=sr//50
d=np.fromfile('phone/full_rec.wav',dtype='<i2',offset=44).astype('float32')/32768
cl=sf.read('runs/full/stream.wav')[0].astype('float32')
plan=json.load(open('runs/full/plan.json'))
def band(x):
    n=len(x)//f*f;fr=x[:n].reshape(-1,f)*np.hanning(f);S=np.abs(np.fft.rfft(fr,axis=1))**2;fq=np.fft.rfftfreq(f,1/sr)
    return 10*np.log10(S[:,(fq>300)&(fq<3000)].sum(1)+1e-9)
ec,er=band(cl),band(d)
def local(e,w=75):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    return e-np.median(sw(np.pad(e,(w,w),mode='edge'),2*w+1),axis=1)
ec,er=local(ec),local(er)
out=np.zeros(int(plan[-1]['window_end']*sr)+sr*5,'float32'); prev=0.0; res=[]
for t in plan:
    a=t['offset']-0.3; b=t['offset']+t['cmd_end']+0.7
    c=ec[int(a/0.02):int(b/0.02)]; n=len(c)
    best=None
    for rad in (3.0,10.0,40.0):
        for k in range(int(-rad/0.02),int(rad/0.02)+1):
            s=int((a+OFF0+prev)/0.02)+k
            if s<0 or s+n>len(er):continue
            cc=np.corrcoef(er[s:s+n],c)[0,1]
            if best is None or cc>best[0]:best=(cc,k)
        if best and best[0]>0.6:break
    ok=bool(best and best[0]>0.6)
    if ok: prev+=best[1]*0.02
    res.append({'order':t['order'],'ok':ok,'corr':round(float(best[0]),2) if best else None,'off':round(prev,2)})
    if ok:
        r0=int((t['offset']-0.3+OFF0+prev)*sr); L=int((t['cmd_end']+t['window_end']-t['offset']-t['cmd_end']+0.3)*sr)  # whole slot
        L=int((t['window_end']-t['offset']+0.3)*sr); seg=d[max(r0,0):r0+L]
        o0=int((t['offset']-0.3)*sr); out[o0:o0+len(seg)]=seg
json.dump(res,open('phone/align.json','w'))
sf.write('phone/aligned.wav',out,sr,subtype='PCM_16')
ok=[r for r in res if r['ok']];print(len(ok),'of',len(res),'trials aligned; last aligned order',max([r['order'] for r in ok]))
print('first lost:',[r['order'] for r in res if not r['ok']][:15])
print('offsets (every 10th):',[(r['order'],r['off'],r['corr']) for r in res[::10]])
