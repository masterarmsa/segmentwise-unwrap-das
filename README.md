# Spatial segment-wise phase unwrapping for distributed acoustic sensing — analysis code

Reference implementation and analysis scripts for the manuscript:

> **Robust and stable phase unwrapping with spatial segment-wise processing for distributed acoustic sensing**
> Zichen Guo, Bin Wang, Haiwei Fu (corresponding author)
> College of Science, Xi'an Shiyou University, Xi'an, Shaanxi 710065, China

---

## 1. What the method does

Conventional DAS processing demodulates each channel against a fixed reference and then
differentiates spatially. Because the differential reference distance is only the spatial
sampling interval **Δz**, the signal is strongly compressed, while a fixed *distant* reference
is fragile — a single reference channel may fall into an interference-fading null.

The method implemented here divides the fibre into short, overlapping segments, unwraps each
segment **locally** with the segment start as its own reference, and thereby extends the
effective reference distance from Δz to **d·Δz**. Unwrapping errors stay confined inside their
own segment instead of propagating along the fibre. A built-in **K-offset grid average**
protects the segment references against interference fading: the segment grid is shifted to
*K* different offsets, each grid is unwrapped independently, and the outputs are averaged
(the vibration signal is coherent across grids; the reference-related noise is not).

## 2. Files

| File | Role |
|---|---|
| `multi_offset.py` | **Core method.** `unwrap_phase_overlap_off()` (overlapping segment unwrapping with an arbitrary grid start offset) and `demod_multi()` (K-offset grid average). Also `apply_mask()` / `extract_pd()`. |
| `real_figs.py` | I/Q → phase front end (`build_Pframes_from_file`, `apply_savgol_filter`), the differential baseline (`demod_baseline`), and spectrum / SNR utilities (`compute_spectrum`, `snr_db`, `find_event`). |
| `m2_gauge_t6.py` | **Reference-distance (gauge-length) experiment.** Uses the paper's SNR estimator `snr_lw()` and sweeps the spatial differential span *G* to show that the SNR gain scales with the reference distance. |
| `m2_fading_cmp.py` | Fading-anchor comparison between the segment-wise method and the baselines. |
| `m2_diff_variants.py` | Variants of the differential baseline (noise/estimator sensitivity). |
| `m2_diff_timediff.py` | Differential baseline with time-domain unwrapping + differencing. |

> `snr_lw(pd, rep)` in `m2_gauge_t6.py` is the estimator used for the SNR values quoted in the
> manuscript: single strongest spectral peak in 5–1000 Hz as signal, mean power of the
> remaining bins (>5 Hz away from the peak) as noise, Hann window.

## 3. Requirements

```
python >= 3.9
numpy
scipy
matplotlib        # only real_figs.py imports it
```

Install with:

```bash
pip install -r requirements.txt
```

## 4. Data format

The scripts read raw I/Q recordings as **interleaved int16**:

- one file = a sequence of frames, frame length **FL = 16384** spatial samples;
- each spatial sample is one **I** followed by one **Q** sample, int16, normalised by 32767;
- the acquisition frame rate (pulse repetition rate) is **REP = 6000** frames/s;
- spatial sampling interval ≈ **0.41 m** per sample (250 MSa/s; used in `m2_gauge_t6.py` as `G * 0.41`);
  the system spatial resolution is `ΔzR` = **10 m** (100 ns probe pulse);
- front-end cut index **FC = 500** and rear mask index **AL = 16000**: `apply_mask()` tapers out the
  samples below `FC` and at or beyond `AL`. The sensing fibre itself occupies **samples 0–15,699**,
  i.e. about **6.4 km** at 0.41 m per sample — the same range as in the manuscript;
- I/Q pre-filter: `apply_savgol_filter(x, IQW, IQP)` with **IQW = 4**, **IQP = 5**; the helper
  evaluates `savgol_filter(x, w, min(p, w - 1))`, so the effective polynomial order is **3**,
  i.e. the four-sample, third-order Savitzky–Golay filter quoted in the manuscript.

Channel indices are **raw, absolute** frame indices and are not re-numbered by the mask: the event
channel of the manuscript is index **12543** (far end, ≈ 5.1 km).

Python snippet used throughout to build a phase matrix `P` of shape `(n_frames, FL)`:

```python
import numpy as np
from real_figs import apply_savgol_filter, FL, IQW, IQP

with open(path, "rb") as f:
    raw = np.frombuffer(f.read(n_frames * FL * 2 * 2), dtype=np.int16)
I = raw[::2].astype(np.float32) / 32767.0
Q = raw[1::2].astype(np.float32) / 32767.0
phase = np.arctan2(apply_savgol_filter(Q, IQW, IQP),
                   apply_savgol_filter(I, IQW, IQP)).astype(np.float32)
P = phase.reshape(n_frames, FL)
```

## 5. Running the scripts

Each analysis script declares the input recording at the top of the file, e.g.

```python
FILE = r"...\6000,50hzsin.bin"
```

**Point that constant at your local copy of the recording, then run it directly:**

```bash
python m2_gauge_t6.py        # reference-distance sweep + proposed method at Nseg = 128
python m2_fading_cmp.py      # fading-anchor comparison
python m2_diff_variants.py   # differential baseline variants
python m2_diff_timediff.py   # time-unwrapped differential baseline
python multi_offset.py <recording.bin> [n_frames] [channel_index]
```

`multi_offset.py` can also be run standalone as a quick demonstration:

```bash
python multi_offset.py 6000,50hzsin.bin 3000 12543
```

## 6. Data availability

The raw Φ-OTDR I/Q recordings are **not** included in this repository. They are several
gigabytes per recording. They are available from the corresponding author on reasonable
request. `multi_offset.py` and the SNR estimator `snr_lw()` run on any recording in the format
described in section 4.

## 7. License

MIT — see `LICENSE`.

## 8. Citation

If you use this code, please cite the manuscript above. If you cite the code package itself,
use its archived DOI (Zenodo).
