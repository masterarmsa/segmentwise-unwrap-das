# -*- coding: utf-8 -*-
"""diff baseline: try unwrap_phase_overlap(num_regions=1) + time-window variants,
to find why our 41 dB vs paper 21.8 dB. Also test rep_rate ambiguity & frames."""
import numpy as np
from scipy import fft
import time, sys
FILE=r"E:\YOLO\Models--YOLO\6000,50hzsin.bin"
FL=16384; EVENT=12543; FRONT_CUT=500; ACTUAL_LEN=16000
def load(path,nbytes):
    with open(path,'rb') as f: return np.frombuffer(f.read(nbytes),dtype=np.int16)
def uw_overlap(phase,seg_len,ov):
    n=len(phase); s_step=seg_len-ov
    if s_step<=0:s_step=1
    out=np.zeros(n); wsum=np.zeros(n); win=np.ones(seg_len)
    if ov>0:
        win[:ov]=np.linspace(0,1,ov); win[-ov:]=np.linspace(1,0,ov)
    st=0
    while st<n:
        en=st+seg_len
        if en>n: en=n; adj=en-st
        else: adj=seg_len
        if adj<=10: break
        uw=np.unwrap(phase[st:en]-phase[st]); aw=win[:adj]
        out[st:en]+=uw*aw; wsum[st:en]+=aw; st+=s_step
    wsum[wsum==0]=1; return out/wsum
def snr(pd,rep,lo=40,hi=60):
    n=len(pd); mag=np.abs(fft.rfft(pd*np.hanning(n))); fr=fft.rfftfreq(n,1.0/rep)
    b=(fr>=5)&(fr<=1000); fb,mb=fr[b],mag[b]; pb=mb**2
    fm=(fb>=lo)&(fb<=hi); f0=fb[fm][np.argmax(mb[fm])] if fm.sum() else fb[np.argmax(mb)]
    sig=np.abs(fb)<=5.0; h=1
    while True:
        if f0*h>1000:break
        sig|=np.abs(fb-f0*h)<=5.0; h+=1
    sp=pb[sig].sum(); nm=~sig; npx=np.median(pb[nm]) if nm.sum() else sp*1e-4
    return 10*np.log10(sp/npx),f0
t0=time.time()
NF=3000
raw=load(FILE,NF*FL*4); I=raw[::2].astype(np.float32)/32767.0; Q=raw[1::2].astype(np.float32)/32767.0
ph=np.arctan2(Q,I)
def collect_diff(mode):
    d=np.empty(NF)
    for i in range(NF):
        fr=ph[i*FL:i*FL+FL].copy(); fr[:FRONT_CUT]=0; fr[ACTUAL_LEN:]=0
        if mode=='rawdiff':            # plain np.unwrap, adjacent
            pu=np.unwrap(fr); d[i]=pu[EVENT]-pu[EVENT-1]
        elif mode=='ov1diff':          # unwrap_phase_overlap(num_regions=1): ov=163
            pu=uw_overlap(fr,FL,int(FL*0.01)); d[i]=pu[EVENT]-pu[EVENT-1]
        elif mode=='ov1_rawchain':     # script: np.diff(Pu)[:,EVENT-1]
            pu=uw_overlap(fr,FL,int(FL*0.01)); dd=np.diff(pu); d[i]=dd[EVENT-1]
    return d
for mode in ['rawdiff','ov1diff','ov1_rawchain']:
    d=collect_diff(mode); pd=d-d.mean()
    for rep in [6000,2000]:
        s,f0=snr(pd,rep)
        print(f"{mode} rep={rep}: diff={s:.2f} dB f0={f0:.1f}",flush=True)
print(f"[{time.time()-t0:.1f}s]")
