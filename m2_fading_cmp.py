# -*- coding: utf-8 -*-
"""
M2 关键对照: 长标距差分(固定远参考) vs proposed K=8 (可移动多参考) — 参考点衰落鲁棒性。

审稿人/文献(如 [10] Wang DDSE 30m 差分)的做法 = 选一个固定差分参考点(远点), 在该点
做相位差。缺点: 参考点若落在深衰落通道(幅度->0), 相位噪声剧增, 整条输出崩。
proposed K=8 = 8 个错开的段网格, 段起点(参考)可移动, 平均后抗衰落。

对照(同一实测 6000,50hzsin.bin, ch12543, W=10000, 论文口径):
  A. 长标距差分 G=97 (参考 12446 正常)      -> 应 ~47.9 dB (等价 proposed)
  B. 长标距差分 G 使参考点落在深衰落通道      -> 预期崩溃 (SNR 大跌 / 峰频漂移)
  C. proposed Nseg=128 K=8                   -> 应稳定 ~45+ dB (含 K=8 平均)
  D. (信息) 该衰落通道做参考时的幅度

SNR 用 snr_lw (Table 6 口径 mean 噪声) + 报告峰频, 崩溃判据=SNR大跌或峰频漂移。
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
    A = np.sqrt(I_f ** 2 + Q_f ** 2).reshape(n_frames, FL).mean(axis=0)
    phase = np.arctan2(Q_f, I_f).astype(np.float32)
    n_avail = min(n_frames, len(phase) // FL)
    P = np.empty((n_avail, FL), dtype=np.float32)
    for i in range(n_avail):
        P[i] = phase[i * FL:(i + 1) * FL]
    return P, n_avail, A

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

P, n, A = build_Pframes_n(FILE, W)
print(f"P={P.shape} 幅度@{IDX}={A[IDX]:.4f}")

# ---- 全局 Itoh ----
Pu = np.unwrap(P, axis=1)
Pu[:, :500] = 0
Pu[:, 16000:] = 0

def long_gauge_snr(G):
    """长标距差分: 参考点 = IDX-G。返回 (SNR, f0, 参考点幅度)。"""
    ref = IDX - G
    d = Pu[:, IDX] - Pu[:, ref]
    pd = np.pad(np.diff(np.unwrap(d)), (0, 1), 'edge')
    s, f0 = snr_lw(pd, REP)
    return s, f0, A[ref]

print("\n=== A/B: 长标距差分 (固定远参考) 扫描参考点位置 ===")
print(f"{'G':>5s}{'参考点':>7s}{'参考幅度':>9s}{'SNR(dB)':>9s}{'峰频(Hz)':>9s}  备注")
# 正常 G 序列 (参考点随机幅度)
for G in [64, 97, 128]:
    s, f0, a = long_gauge_snr(G)
    note = "正常参考"
    print(f"{G:>5d}{IDX-G:>7d}{a:>9.4f}{s:>9.2f}{f0:>9.1f}  {note}")
# 衰落参考: 找事件附近深衰落通道 (11000..12800 幅度最低的几个)
near = np.arange(11000, 12800)
dark_idx = near[A[near].argsort()][:6]
for dk in dark_idx:
    G = IDX - dk
    if G <= 0: continue
    s, f0, a = long_gauge_snr(G)
    note = "深衰落参考!" if a < 0.15 * np.median(A) else ""
    print(f"{G:>5d}{dk:>7d}{a:>9.4f}{s:>9.2f}{f0:>9.1f}  {note}")

# ---- C: proposed Nseg=128 K=8 (默认实现) ----
_, pd8 = MO.demod_multi(P, IDX, 128, 8)
s8, f8 = snr_lw(pd8, REP)
print(f"\n=== C: proposed Nseg=128 K=8 = {s8:.2f} dB @ {f8:.1f} Hz")

# ---- D: proposed K=1 同衰落参考(单网格段起点恰为衰落点) 对照 ----
# 单网格时若段起点落深衰落通道: 用 off 使事件段起点 = dk(最暗)
seg_len = FL // 128
for dk in dark_idx[:3]:
    # 网格起点 off 使段起点序列 s_m = off + m*(seg_len-ov) 覆盖 dk
    # 简化: unwrap_phase_overlap_off(off) 从 off 起步, 直接看 dk 是否落在事件段起点
    for off in range(0, seg_len, 7):  # 粗扫 7 个偏移近似 K 网格
        pass
    # 直接构造: off 使事件点所在段的起点 = dk (若可)
    # 事件段起点 kS = off + m*step, 要 kS==dk 则 off = dk - m*step; 找合法 off
    s_step = seg_len - int(seg_len * MO.OVL)
    cand = []
    for m in range(200):
        offc = dk - m * s_step
        if 0 <= offc < seg_len:
            cand.append(offc)
            break
    if cand:
        c1, pd1 = None, None
        # 单网格 K=1 指定 off
        ov1 = int(seg_len * MO.OVL)
        Pu1 = MO.unwrap_phase_overlap_off(P, seg_len, ov1, cand[0])
        Pu1 = MO.apply_mask(Pu1)
        _, pd1 = MO.extract_pd(Pu1, IDX)
        s1, f1 = snr_lw(pd1, REP)
        print(f"D: K=1 段起点@{dk} (off={cand[0]}): SNR={s1:.2f} dB @ {f1:.1f} Hz")

print("\nDONE")
