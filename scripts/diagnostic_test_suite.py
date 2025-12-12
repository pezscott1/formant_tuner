import os, json, argparse
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

from formant_utils import (
    is_plausible_formants,
    estimate_formants_lpc,
    lpc_envelope_peaks,
    smoothed_spectrum_peaks,
)

def analyze_file(path, mid_len_seconds=1.5, frame_ms=60):
    y, sr = sf.read(path)
    total = len(y)
    mid_len = min(int(mid_len_seconds * sr), total)
    mid_start = max(0, total//2 - mid_len//2)
    mid = y[mid_start:mid_start+mid_len].astype(float)
    frame_len = max(32, int(sr * (frame_ms/1000.0)))
    center_frame = mid[len(mid)//2 - frame_len//2 : len(mid)//2 + frame_len//2]

    f1, f2 = estimate_formants_lpc(center_frame, sr)
    ok, reason = is_plausible_formants(f1, f2)
    return {"path": path, "sr": sr, "chosen_f1": f1, "chosen_f2": f2,
            "plausible": ok, "reason": reason}

def run_batch(input_dir, out_csv="report.csv"):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")]
    rows = [analyze_file(p) for p in files]
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    fails = sum(1 for r in rows if not r["plausible"])
    print(f"Analyzed {len(rows)} files: {fails} failures")
    return 0 if fails == 0 else 2

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", default="report.csv")
    args = p.parse_args()
    rc = run_batch(args.input_dir, out_csv=args.out)
    raise SystemExit(rc)