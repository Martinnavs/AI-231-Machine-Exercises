"""Mel-band channel map between the clean in-Pi audio and what the mic recorded (phone speaker -> air -> USB mic)."""
import numpy as np
sr=16000; NFFT=512; HOP=128; NM=40
def mel(f): return 2595*np.log10(1+f/700)
def imel(m): return 700*(10**(m/2595)-1)
def fb():
    f=np.fft.rfftfreq(NFFT,1/sr); pts=imel(np.linspace(mel(50),mel(7900),NM+2)); W=np.zeros((NM,len(f)))
    for i in range(NM):
        a,b,c=pts[i:i+3]; W[i]=np.clip(np.minimum((f-a)/(b-a),(c-f)/(c-b)),0,None)
    return W,pts[1:-1]
W,CENT=fb()
def stft(x):
    n=1+(len(x)-NFFT)//HOP; idx=np.arange(NFFT)[None]+HOP*np.arange(n)[:,None]
    return np.fft.rfft(x[idx]*np.hanning(NFFT),axis=1)
def istft(S,n):
    y=np.zeros(n);w=np.zeros(n);win=np.hanning(NFFT)
    for i,s in enumerate(S):
        fr=np.fft.irfft(s,NFFT)*win; y[i*HOP:i*HOP+NFFT]+=fr; w[i*HOP:i*HOP+NFFT]+=win**2
    return (y/np.maximum(w,1e-8)).astype('float32')
def melpow(S): return (np.abs(S)**2)@W.T
def fit(clean,rec,top=0.3):
    """per-mel-band gain (dB) so rec*gain ~ clean, from the loudest `top` fraction of frames."""
    Sc,Sr=stft(clean),stft(rec); n=min(len(Sc),len(Sr)); Mc,Mr=melpow(Sc[:n]),melpow(Sr[:n])
    e=Mc.sum(1); sel=e>=np.quantile(e,1-top)
    return 10*np.log10(Mc[sel].mean(0)+1e-12)-10*np.log10(Mr[sel].mean(0)+1e-12), (10*np.log10(Mc[sel].mean(0)+1e-12), 10*np.log10(Mr[sel].mean(0)+1e-12))
def apply(rec,gdb,maxboost=25):
    S=stft(rec); f=np.fft.rfftfreq(NFFT,1/sr); g=np.interp(f,CENT,np.clip(gdb,-40,maxboost))
    return istft(S*10**(g/20),len(rec))
