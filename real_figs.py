# -*- coding: utf-8 -*-
"""
真实 DAS 数据 -> 分段法 vs 差分法 对比图（v14 同款 SCI 排版）
复用 lunwen10.py 的读 .bin + 解缠逻辑（unwrap_phase_overlap / apply_mask），
SNR 用论文旧口径 snr_db(curve)（与 fig18-22 一致）。

数据格式（lunwen10.py / compare_real_vs_sim.py 确认）：
  .bin = int16，I/Q 交错：I=DATA[::2], Q=DATA[1::2]
  相位 = arctan2(Q, I)；FL=16384 空间点/帧；REP=6000 帧/秒
  FC=500(前端截除), AL=16000(7km光纤), num_regions=32
"""
import os, importlib.util, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.fft as fft

# ---------------- 载入旧模拟器，仅借用 unwrap_phase_overlap / apply_savgol_filter（算法一致） ----------------
SPEC = r'C:\Users\gzc\Desktop\8\das_sim_demo.py'
spec = importlib.util.spec_from_file_location("das_sim", SPEC)
das = importlib.util.module_from_spec(spec)
spec.loader.exec_module(das)

import multi_offset as MO   # 多偏移网格平均（方法升级 B，K=8 默认）

# ---------------- 真实系统参数 ----------------
FL = 16384            # 每帧空间点数
REP = 6000            # 脉冲重复频率（帧/秒）
FC = 500              # 前端截取索引
AL = 16000            # 实际光纤长度索引（7km）
TW = 100              # 截断过渡带
IQW = 4               # Savgol 窗宽
IQP = 5               # Savgol 阶数
NUM_REGIONS = 32      # 分段数（proposed 默认）
OVL = 0.01            # 重叠比例

OUT = r'C:\Users\gzc\Desktop\8\reviewer2_v14_real'
os.makedirs(OUT, exist_ok=True)

# ---------------- 论文样式（对齐 lunwen10.1 / v14） ----------------
try:
    plt.style.use('seaborn-v0_8-paper')
except Exception:
    pass
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Palatino Linotype', 'SimHei', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 3, 'ytick.minor.size': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'legend.frameon': True, 'legend.fontsize': 9,
    'legend.handlelength': 1.5, 'legend.handletextpad': 0.4,
    'figure.dpi': 150, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
C_SEG = '#1f5fb4'
C_DIFF = (0.20, 0.20, 0.20)
LS_DIFF = (0, (4, 2))
FN = 'Times New Roman'
LW_TS = 1.3
LW_SP = 1.1

# ---------------- 读真实 .bin（lunwen10.load_partial_data 同口径） ----------------
def load_partial_data(file_path, num_segments, frame_len, step):
    needed = (num_segments * step + frame_len) * 2 * 2
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(needed), dtype=np.int16)
    n_frames = len(data) // (2 * step)
    return data, min(num_segments, n_frames)

def apply_savgol_filter(s, w=5, p=3):
    if w <= 1:
        return s
    if w % 2 == 0:
        w += 1
    return savgol_filter(s, w, min(p, w - 1))

def build_Pframes_from_file(file_path, num_segments=3000):
    DATA, nseg = load_partial_data(file_path, num_segments, FL, FL)
    I = DATA[::2].astype(np.float32) / 32767.0
    Q = DATA[1::2].astype(np.float32) / 32767.0
    I_f = apply_savgol_filter(I, IQW, IQP)
    Q_f = apply_savgol_filter(Q, IQW, IQP)
    phase_raw = np.arctan2(Q_f, I_f).astype(np.float32)
    n_frames = phase_raw.shape[0] // FL
    P_frames = np.empty((n_frames, FL), dtype=np.float32)
    for i in range(n_frames):
        P_frames[i] = phase_raw[i * FL:(i + 1) * FL]
    return P_frames

# ---------------- unwrap_phase_overlap（抄 lunwen10，算法一致） ----------------
def unwrap_phase_overlap(phase_frames, seg_len=256, ov_len=128):
    n_f, flen = phase_frames.shape
    s_step = seg_len - ov_len
    if s_step <= 0:
        s_step = 1
    n_segs = (flen - ov_len) // s_step + 1
    out = np.zeros((n_f, flen), dtype=np.float32)
    wsum = np.zeros_like(out)
    win = np.ones(seg_len, dtype=np.float32)
    if ov_len > 0:
        win[:ov_len] = np.linspace(0, 1, ov_len)
        win[-ov_len:] = np.linspace(1, 0, ov_len)
    for fi in range(n_f):
        for si in range(n_segs):
            st = si * s_step
            en = st + seg_len
            if en > flen:
                en = flen
                adj = en - st
                if adj <= 10:
                    continue
            else:
                adj = seg_len
            seg = phase_frames[fi, st:en]
            uw = np.unwrap(seg - seg[0])
            if adj == seg_len:
                out[fi, st:en] += uw * win
                wsum[fi, st:en] += win
            else:
                awin = win[:adj]
                out[fi, st:en] += uw * awin
                wsum[fi, st:en] += awin
    wsum[wsum == 0] = 1.0
    return out / wsum

def apply_mask(Pu):
    front_mask = np.arange(Pu.shape[1]) < FC
    rear_mask = np.arange(Pu.shape[1]) >= AL
    Pu[:, front_mask] = 0
    Pu[:, rear_mask] = 0
    if FC > TW:
        ft = FC - TW
        w = np.linspace(0, 1, TW)
        for i in range(Pu.shape[0]):
            Pu[i, ft:FC] *= w
    if AL > TW:
        rt = AL - TW
        w = np.linspace(1, 0, TW)
        for i in range(Pu.shape[0]):
            Pu[i, rt:AL] *= w
    return Pu

def extract_pd(Pu, idx):
    ph = Pu[:, idx]
    curve = np.unwrap(ph)
    pd = np.diff(curve)
    pd = np.pad(pd, (0, 1), 'edge')
    return curve - curve[0], pd

def demod_proposed(P_frames, idx, nregions=NUM_REGIONS, K=8):
    """本文方法：空间分块重叠解缠 + 多偏移网格平均（K组错位网格，输出平均）。
    单段参考若落在低幅度(相位衰落)通道会污染该段；信号跨网格相干、参考噪声
    不相干，平均即消 → 参考点落位稳健。"""
    return MO.demod_multi(P_frames, idx, nregions, K)

def demod_baseline(P_frames, idx):
    Pu_full = np.unwrap(P_frames, axis=1)
    Pu_full = apply_mask(Pu_full)
    return extract_pd(Pu_full, idx)

def snr_db(curve, freq, wf='sine'):
    t = np.arange(len(curve)) / REP
    cols = [np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)]
    if wf in ('square', 'triangle'):
        for h in (3, 5, 7):
            cols += [np.sin(2 * np.pi * h * freq * t), np.cos(2 * np.pi * h * freq * t)]
    H = np.array(cols).T
    coef, *_ = np.linalg.lstsq(H, curve, rcond=None)
    w = H @ coef
    A = np.sqrt(np.mean(w ** 2))
    sig = np.std(curve - w)
    return float(20 * np.log10(A / sig)) if sig > 0 else 99.0

def compute_spectrum(pd):
    n_s = len(pd)
    win = np.hanning(n_s)
    fft_res = fft.rfft(pd * win)
    mag = np.abs(fft_res)
    freq = fft.rfftfreq(n_s, d=1 / REP)
    return freq, mag

def find_event(P_frames, target_freq=100.0, candidates=None):
    """自动定位振动点：排除<20Hz低频漂移，以目标频带(默认100Hz)功率最大处为振动点。
    返回 (best_idx, best_freq, rows) —— rows 为诊断表。"""
    if candidates is None:
        candidates = list(range(1000, 15000, 1000))
    rows = []
    for idx in candidates:
        if idx < 1 or idx >= FL - 1:
            continue
        c, pd = demod_proposed(P_frames, idx)
        fr, mag = compute_spectrum(pd)
        # 主频（排除<20Hz低频漂移伪峰）
        band = (fr >= 20) & (fr <= 1000)
        pk = np.argmax(mag[band])
        freq = fr[band][pk]
        snr = snr_db(c, freq, 'sine')
        # 目标频带(±10Hz)功率：分段法 vs 差分法
        band_t = (fr >= target_freq - 10) & (fr <= target_freq + 10)
        p_t = mag[band_t].max() if band_t.any() else 0.0
        _, pd_d = demod_baseline(P_frames, idx)
        _, mag_d = compute_spectrum(pd_d)
        p_t_d = mag_d[band_t].max() if band_t.any() else 0.0
        rows.append((idx, float(freq), float(snr), float(p_t), float(p_t_d)))
    # 取目标频带功率最大处
    best = max(rows, key=lambda r: r[3])
    best_idx = best[0]
    # 在最佳索引重算主频（用于 LS 拟合/标注）
    c, pd = demod_proposed(P_frames, best_idx)
    fr, mag = compute_spectrum(pd)
    band = (fr >= 20) & (fr <= 1000)
    freq = float(fr[band][np.argmax(mag[band])])
    return best_idx, freq, rows

def plot_pair(idx, freq, P_frames, fname, nregions=NUM_REGIONS):
    c_seg, pd_seg = demod_proposed(P_frames, idx, nregions)
    c_base, pd_base = demod_baseline(P_frames, idx)
    # 真实数据曲线受低频环境漂移主导（共模，两法+平均都无法去除），
    # 论文口径改在"相位差 pd"域评估 SNR（即图中实际绘制的量）
    snr_seg = snr_db(pd_seg, freq, 'sine')
    snr_base = snr_db(pd_base, freq, 'sine')
    snr_c_seg = snr_db(c_seg, freq, 'sine')
    snr_c_base = snr_db(c_base, freq, 'sine')
    print(f"  [idx {idx} @ {freq:.1f}Hz] 分段SNR(pd)={snr_seg:.1f}dB  差分SNR(pd)={snr_base:.1f}dB  gap={snr_seg-snr_base:.1f}dB"
          f"   | 曲线口径: {snr_c_seg:.1f}/{snr_c_base:.1f}")
    t = np.arange(len(pd_seg)) / REP
    mask = t <= 0.05
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.0))
    # 时域
    ax = axs[0]
    ax.plot(t[mask], pd_seg[mask], color=C_SEG, lw=LW_TS, label='Segmented (proposed)')
    ax.plot(t[mask], pd_base[mask], color=C_DIFF, lw=LW_TS - 0.2, linestyle=LS_DIFF, label='Differential (baseline)')
    ax.set_xlim(0, 0.05)
    ax.set_xlabel('Time (s)', fontfamily=FN)
    ax.set_ylabel('Phase differential (rad)', fontfamily=FN)
    ax.set_title(f'(a) Phase differential @ idx {idx}', fontsize=11, fontweight='bold', fontfamily=FN)
    ax.tick_params(direction='in', which='both')
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax.minorticks_on()
    ax.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
    ax.text(0.98, 0.92, f'SNR(pd): seg={snr_seg:.1f}  diff={snr_base:.1f} dB',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.95))
    # 频谱
    ax = axs[1]
    fr_seg, mag_seg = compute_spectrum(pd_seg)
    fr_base, mag_base = compute_spectrum(pd_base)
    band = (fr_seg >= 5) & (fr_seg <= 1000)
    ax.plot(fr_seg[band], mag_seg[band], color=C_SEG, lw=LW_SP, label='Segmented')
    ax.plot(fr_base[band], mag_base[band], color=C_DIFF, lw=LW_SP - 0.2, linestyle=LS_DIFF, label='Differential')
    ax.set_xlim(5, 1000)
    ax.set_xlabel('Frequency (Hz)', fontfamily=FN)
    ax.set_ylabel('Magnitude (a.u.)', fontfamily=FN)
    ax.set_title(f'(b) Spectrum', fontsize=11, fontweight='bold', fontfamily=FN)
    ax.tick_params(direction='in', which='both')
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax.minorticks_on()
    ax.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')
    fig.tight_layout()
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  写图: {fname}\n")
    return snr_seg, snr_base

if __name__ == '__main__':
    FILE = sys.argv[1] if len(sys.argv) > 1 else r'E:\YOLO\Models--YOLO\6000iq100hz正弦.bin'
    NFR = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 12543   # 用户指定的真实振动点
    print(f"读取真实数据: {FILE}  (前 {NFR} 帧)  固定振动点 idx={IDX}")
    P_frames = build_Pframes_from_file(FILE, NFR)
    print(f"  P_frames shape = {P_frames.shape}, REP={REP}")

    # ---- 直接分析固定振动点 12543：分段法 vs 差分法 ----
    c_seg, pd_seg = demod_proposed(P_frames, IDX)
    c_base, pd_base = demod_baseline(P_frames, IDX)
    fr_s, mag_s = compute_spectrum(pd_seg)
    fr_b, mag_b = compute_spectrum(pd_base)
    band = (fr_s >= 20) & (fr_s <= 1000)
    f_seg = float(fr_s[band][np.argmax(mag_s[band])])
    f_base = float(fr_b[band][np.argmax(mag_b[band])])
    snr_seg_c = snr_db(c_seg, f_seg, 'sine')
    snr_base_c = snr_db(c_base, f_base, 'sine')
    snr_seg_p = snr_db(pd_seg, f_seg, 'sine')
    snr_base_p = snr_db(pd_base, f_base, 'sine')
    # 100Hz 目标频带功率（验证振动是否真正恢复）
    bt = (fr_s >= 90) & (fr_s <= 110)
    P100_seg = float(mag_s[bt].max()) if bt.any() else 0.0
    P100_base = float(mag_b[bt].max()) if bt.any() else 0.0
    print(f"\n[idx {IDX}] 分段法: 主频={f_seg:.1f}Hz  曲线SNR={snr_seg_c:.1f}dB  pdSNR={snr_seg_p:.1f}dB  100Hz带峰={P100_seg:.3e}")
    print(f"[idx {IDX}] 差分法: 主频={f_base:.1f}Hz  曲线SNR={snr_base_c:.1f}dB  pdSNR={snr_base_p:.1f}dB  100Hz带峰={P100_base:.3e}")

    plot_pair(IDX, f_seg, P_frames, os.path.join(OUT, 'real_proto_fig.png'))
    print("DONE")
