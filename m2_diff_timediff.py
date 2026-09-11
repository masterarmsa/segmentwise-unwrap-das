# -*- coding: utf-8 -*-
"""Apply the SAME time processing as proposed (np.unwrap + np.diff) to the diff
baseline time series, to see if 41 dB drops toward paper 21.8 dB. 3000 frames."""
import numpy as np
from scipy import fft
import time
FILE=r"E:\YOLO\Models--YOLO\6000,50hzsin.bin"
FL=16384; EVENT=12543; REP=6000; NF=3000; FRONT_CUT=500; ACTUAL_LEN=16000
with open(FILE,'rb') as f: raw=np.frombuffer(f.read(NF*FL*4),dtype=np.int16)
I=raw[::2].astype(np.float32)/32767.0; Q=raw[1::2].astype(np.float32)/32767.0
ph=np.arctan2(Q,I)
d=np.empty(NF)
for i in range(NF):
    fr=ph[i*FL:i*FL+FL].copy(); fr[:FRONT_CUT]=0; fr[ACTUAL_LEN:]=0
    pu=np.unwrap(fr); d[i]=pu[EVENT]-pu[EVENT-1]
def snr(pd,rep=6000):
    n=len(pd); mag=np.abs(fft.rfft(pd*np.hanning(n))); fr=fft.rfftfreq(n,1.0/rep)
    b=(fr>=5)&(fr<=1000); fb,mb=fr[b],mag[b]; pb=mb**2
    fm=(fb>=40)&(fb<=60); f0=fb[fm][np.argmax(mb[fm])]
    sig=np.abs(fb)<=5.0; h=1
    while True:
        if f0*h>1000:break
        sig|=np.abs(fb-f0*h)<=5.0; h+=1
    sp=pb[sig].sum(); nm=~sig; npx=np.median(pb[nm])
    return 10*np.log10(sp/npx)
pd_raw=d-d.mean()
print("diff, time-DC removed          :", round(snr(pd_raw),2))
# apply proposed-style time unwrap + diff
pc=np.unwrap(d); pd2=np.diff(pc); pd2=np.pad(pd2,(0,1),'edge'); pd2=pd2-pd2.mean()
print("diff, + time unwrap&diff (prop):", round(snr(pd2),2))
