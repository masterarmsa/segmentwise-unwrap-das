# -*- coding: utf-8 -*-
"""
M2 标距匹配 — 在 Table 6 精确口径下验证审稿人指控。

Table 6 差分基线 21.8 dB = base_pd(全局 Itoh -> 相邻空间差分 -> 时间unwrap+diff) + snr_lw
proposed 47.9 dB = MO.demod_multi(P,12543,128,K=1) + snr_lw，W=10000 帧。

审稿人 M2: proposed 相对差分基线的增益主要来自"参考距离/标距从相邻 1 样点拉长到
Nseg=128 的段长 d"（信号相干积分），而非分段解缠机制本身。

验证方法：对全局 Itoh Pu(z)，把空间差分的标距从 G=1 拉长到 G=d（Pu[:,idx]-Pu[:,idx-G]），
在完全相同的时间unwrap+diff + snr_lw 口径下算 SNR，看是否随 G 增长并逼近 proposed 47.9。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.fft as fft
from real_figs import FL, IQW, IQP, apply_savgol_filter
import multi_offset as MO

FILE = r"C:\Users\gzc\WorkBuddy\2026-07-29-12-00-31\data_12543\6000,50hzsin.bin"
IDX = 12543
REP = 6000
W = 10000

def build_Pframes_n(file_path, n_frames):
    need = n_frames * FL * 2 * 2
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(need), dtype=np.int16)
    I = data[::2].astype(np.float32) / 32767.0
    Q = data[1::2].astype(np.float32) / 32767.0
    I_f = apply_savgol_filter(I, IQW, IQP)
    Q_f = apply_savgol_filter(Q, IQW, IQP)
    phase = np.arctan2(Q_f, I_f).astype(np.float32)
    n_avail = min(n_frames, len(phase) // FL)
    P = np.empty((n_avail, FL), dtype=np.float32)
    for i in range(n_avail):
        P[i] = phase[i * FL:(i + 1) * FL]
    return P, n_avail

def snr_lw(pd, rep, vmin=5.0, vmax=1000.0):
    n = len(pd)
    mag = np.abs(fft.rfft(pd * np.hanning(n)))
    freq = fft.rfftfreq(n, d=1.0 / rep)
    band = (freq >= vmin) & (freq <= vmax)
    fb, mb = freq[band], mag[band]
    pk = np.argmax(mb)
    notch = np.abs(fb - fb[pk]) > 5
    nz = np.mean(mb[notch] ** 2) if notch.sum() > 0 else mb[pk] ** 2 / 100
    return 10 * np.log10(mb[pk] ** 2 / max(nz, 1e-12)), float(fb[pk])

def gauge_pd(Pu, idx, G):
    """全局 Itoh 后 span=G 样点的空间差分 + 时间 unwrap+diff (与 base_pd 同型)。"""
    d = Pu[:, idx] - Pu[:, idx - G]
    return np.pad(np.diff(np.unwrap(d)), (0, 1), 'edge')

P, n_avail = build_Pframes_n(FILE, W)
print(f"P_frames={P.shape} (实际{n_avail}帧)")

# ---- proposed (Table 6 口径, Nseg=128 K=1) ----
_, pd128 = MO.demod_multi(P, IDX, 128, 1)
s128, f128 = snr_lw(pd128, REP)
print(f"proposed Nseg=128 (K=1) = {s128:.2f} dB @ {f128:.1f} Hz   [论文 Table6: 47.9]")

# ---- 全局 Itoh Pu ----
Pu = np.unwrap(P, axis=1)
Pu[:, :500] = 0
Pu[:, 16000:] = 0

print(f"\n{'G':>5s}{'参考距离d(m)':>12s}{'SNR(dB)':>10s}{'峰频(Hz)':>10s}")
results = []
for G in [1, 2, 8, 24, 32, 64, 97, 128, 256]:
    d_m = G * 0.41
    s, f0 = snr_lw(gauge_pd(Pu, IDX, G), REP)
    results.append((G, d_m, s, f0))
    print(f"{G:>5d}{d_m:>12.2f}{s:>10.2f}{f0:>10.1f}")

print("\n=== 汇总 ===")
print(f"proposed Nseg=128 = {s128:.2f} dB (参考距离 d≈{128*0.41:.1f} m, 段长128样点)")
print("审稿人 M2 指控: proposed 相对差分基线(21.8) 的增益在标距拉长时是否被滑动差分复现？")
