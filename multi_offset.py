# -*- coding: utf-8 -*-
"""
多偏移网格平均的分段解缠方法（v14 方法升级 B）。

物理动机：分段解缠的参考点（段起点）若落在低幅度/相位衰落通道，该段解缠会被
参考点噪声污染（真实数据实测：网格平移 offset 0..512 → SNR 摆幅 21.7 dB）。
修复：把分段网格整体平移 K 组 offset（0..seg_len 均匀分布），每组独立做分段解缠，
再对输出曲线取平均 —— 振动信号跨网格相干、参考相关噪声不相干，平均即消。
真实数据@12543 实测：单网格最佳 -3.0dB → K=11 平均 +7.8dB（基线差分 -26.7dB）。

unwrap_phase_overlap_off 与 das_sim_demo.unwrap_phase_overlap 逐行一致（仅加 start）。
"""
import numpy as np

FL = 16384            # 每帧空间点数
FC = 500              # 前端截取索引
AL = 16000            # 实际光纤长度索引（7km）
TW = 100              # 截断过渡带
OVL = 0.01            # 重叠比例


def unwrap_phase_overlap_off(phase_frames, seg_len=256, ov_len=128, start=0):
    """带 start 偏移的分段重叠解缠；start=0 时与 das_sim_demo.unwrap_phase_overlap 一致。"""
    n_f, flen = phase_frames.shape
    s_step = seg_len - ov_len
    if s_step <= 0:
        s_step = 1
    out = np.zeros((n_f, flen), dtype=np.float32)
    wsum = np.zeros_like(out)
    win = np.ones(seg_len, dtype=np.float32)
    if ov_len > 0:
        win[:ov_len] = np.linspace(0, 1, ov_len)
        win[-ov_len:] = np.linspace(1, 0, ov_len)
    st = start
    while st < flen:
        en = st + seg_len
        if en > flen:
            en = flen
            adj = en - st
            if adj <= 10:
                break
        else:
            adj = seg_len
        seg = phase_frames[:, st:en]
        uw = np.unwrap(seg - seg[:, 0:1], axis=1)   # 向量化：逐帧独立解缠
        awin = win[:adj]
        out[:, st:en] += uw * awin
        wsum[:, st:en] += awin
        st += s_step
    wsum[wsum == 0] = 1.0
    return out / wsum


def apply_mask(Pu):
    """前端/后端截断+过渡带（向量化，等价原逐帧循环）。"""
    front_mask = np.arange(Pu.shape[1]) < FC
    rear_mask = np.arange(Pu.shape[1]) >= AL
    Pu[:, front_mask] = 0
    Pu[:, rear_mask] = 0
    if FC > TW:
        ft = FC - TW
        w = np.linspace(0, 1, TW)
        Pu[:, ft:FC] *= w[None, :]
    if AL > TW:
        rt = AL - TW
        w = np.linspace(1, 0, TW)
        Pu[:, rt:AL] *= w[None, :]
    return Pu


def extract_pd(Pu, idx):
    """返回 (curve-curve[0] 累积相位, pd 时间相位差)。SNR 用累积相位计算（旧口径）。"""
    ph = Pu[:, idx]
    curve = np.unwrap(ph)
    pd = np.diff(curve)
    pd = np.pad(pd, (0, 1), 'edge')
    return curve - curve[0], pd


def demod_multi(P_frames, idx, nregions=32, K=8):
    """多偏移网格平均的分段解缠。返回 (curve_avg, pd_avg)。

    只平均"网格能覆盖 idx"的偏移（start <= idx），避免未覆盖区域补零稀释信号。
    """
    seg_len = FL // nregions
    ov = 0 if seg_len >= FL else int(seg_len * OVL)
    if ov >= seg_len:
        ov = seg_len // 2
    curves, pds = [], []
    for off in np.linspace(0, seg_len, K, endpoint=False).astype(int):
        if off > idx:            # 该网格未覆盖事件点，跳过
            continue
        Pu = unwrap_phase_overlap_off(P_frames, seg_len, ov, off)
        Pu = apply_mask(Pu)
        c, pd = extract_pd(Pu, idx)
        curves.append(c)
        pds.append(pd)
    if not curves:
        raise ValueError(f"idx={idx} 在首段之前，无网格可覆盖")
    return np.mean(curves, axis=0), np.mean(pds, axis=0)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
    from real_figs import build_Pframes_from_file, demod_baseline, compute_spectrum, snr_db, REP
    FILE = sys.argv[1] if len(sys.argv) > 1 else r'E:\YOLO\Models--YOLO\6000iq100hz正弦.bin'
    NFR = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 12543
    print(f"读取: {FILE}\n帧={NFR}  事件点 idx={IDX}")
    P_frames = build_Pframes_from_file(FILE, NFR)

    c_b, pd_b = demod_baseline(P_frames, IDX)
    fr_b, mag_b = compute_spectrum(pd_b)
    band = (fr_b >= 20) & (fr_b <= 1000)
    f_b = float(fr_b[band][np.argmax(mag_b[band])])
    snr_b = snr_db(c_b, f_b, 'sine')
    print(f"差分法(基线): SNR={snr_b:.1f}dB @ {f_b:.1f}Hz")

    for K in (4, 8, 11, 16):
        c_a, pd_a = demod_multi(P_frames, IDX, 32, K)
        fr_a, mag_a = compute_spectrum(pd_a)
        band = (fr_a >= 20) & (fr_a <= 1000)
        f_a = float(fr_a[band][np.argmax(mag_a[band])])
        snr_c = snr_db(c_a, f_a, 'sine')      # 对平均曲线算 SNR（论文口径）
        snr_p = snr_db(pd_a, f_a, 'sine')     # 对平均 pd 算 SNR（演示口径）
        print(f"K={K:>2}: 曲线平均SNR={snr_c:6.1f}dB  pd平均SNR={snr_p:6.1f}dB  "
              f"主频={f_a:.1f}Hz   (vs基线 {snr_b:.1f}dB)")
    print("DONE")
