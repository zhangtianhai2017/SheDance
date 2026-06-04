#!/usr/bin/env python3
"""Assess kinematic beat detectability from the dance motion (energy autocorrelation -> tempo).
Decides whether motion-based beat is viable (when no audio)."""
import numpy as np
import render_core as rc
c = np.load(rc.CACHE, allow_pickle=True)
qvel = np.asarray(c["qvel"], float); freq = float(c["frequency"])
e = np.linalg.norm(qvel, axis=1)
e = np.convolve(e, np.ones(5) / 5, mode="same"); e = e - e.mean()
ac = np.correlate(e, e, "full")[len(e) - 1:]
lo, hi = int(0.30 * freq), int(1.2 * freq)          # 50-200 BPM plausible
period = lo + int(np.argmax(ac[lo:hi])); bpm = 60 * freq / period
print(f"frames={len(e)} dur={len(e)/freq:.1f}s")
print(f"kinematic tempo estimate: {bpm:.1f} BPM  (beat period {period/freq:.2f}s)")
dd = np.diff(np.sign(np.diff(e)))
peaks = [p + 1 for p in np.where(dd < 0)[0] if e[p + 1] > e.std() * 0.5]
print(f"energy local maxima (candidate beats): {len(peaks)} = {len(peaks)/(len(e)/freq)*60:.0f}/min")
print(f"autocorr peak strength: {ac[period]/ac[0]:.2f} (higher=clearer periodicity)")
