"""Causal front-end level control: slow-attack/fast-release style AGC with a noise gate. Frame-by-frame, no look-ahead."""
import numpy as np
sr=16000
def agc(x,target_dbfs=-22.0,max_gain_db=30.0,gate_margin_db=8.0,frame=320,up_ms=250,down_ms=30):
    n=len(x)//frame; y=np.zeros_like(x); g=0.0; floor=None
    a_up=1-np.exp(-frame/sr/(up_ms/1000)); a_dn=1-np.exp(-frame/sr/(down_ms/1000))
    for i in range(n):
        s=x[i*frame:(i+1)*frame]; lvl=20*np.log10(np.sqrt((s**2).mean())+1e-9)
        floor=lvl if floor is None else (floor+0.002*(lvl-floor) if lvl>floor else floor+0.05*(lvl-floor))   # slow rise, fast fall
        if lvl>floor+gate_margin_db:                      # speech-like frame: steer gain toward the target
            want=min(max_gain_db,target_dbfs-lvl)
            g+= (a_dn if want<g else a_up)*(want-g)
        # else: hold the current gain (noise gate)
        y[i*frame:(i+1)*frame]=np.clip(s*10**(g/20),-1,1)
    y[n*frame:]=x[n*frame:]*10**(g/20)
    return y
