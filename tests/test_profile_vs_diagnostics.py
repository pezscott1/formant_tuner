import os, json, math, statistics, csv, re, unittest

PROFILE_PATH = os.path.join("profiles", "user1_profile.json")
CSV_PATH = "report.csv"

F1_RANGE = (50.0, 1200.0)
F2_RANGE = (400.0, 5000.0)
MIN_SEP = 200.0
MISMATCH_TOLERANCE_HZ = 100.0

def is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))

def load_profile(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_profile(profile):
    failures = []
    for v, rec in sorted(profile.items()):
        f1 = rec.get("f1"); f2 = rec.get("f2"); retries = rec.get("retries")
        if f1 is not None and not is_finite(f1):
            failures.append(f"{v}: f1 not finite ({f1})")
        if f2 is not None and not is_finite(f2):
            failures.append(f"{v}: f2 not finite ({f2})")
        if retries is None or not isinstance(retries, int):
            failures.append(f"{v}: retries missing or not int ({retries})")
        if f1 is not None and f2 is not None:
            if float(f1) > float(f2):
                failures.append(f"{v}: f1 > f2 ({f1} > {f2})")
            if not (F1_RANGE[0] <= float(f1) <= F1_RANGE[1]):
                failures.append(f"{v}: f1 out of range ({f1})")
            if not (F2_RANGE[0] <= float(f2) <= F2_RANGE[1]):
                failures.append(f"{v}: f2 out of range ({f2})")
            if (float(f2) - float(f1)) < MIN_SEP:
                failures.append(f"{v}: f2-f1 too small ({float(f2)-float(f1):.1f} Hz)")
    return failures

def csv_medians_by_vowel(csv_path):
    if not os.path.exists(csv_path):
        return {}
    per_vowel = {}
    with open(csv_path, newline="", encoding="utf-8") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            m = re.search(r"_([aeiou])_", row.get("file", ""))
            if not m:
                continue
            v = m.group(1)
            try:
                f1 = float(row.get("chosen_f1") or "nan")
                f2 = float(row.get("chosen_f2") or "nan")
            except Exception:
                continue
            per_vowel.setdefault(v, {"f1": [], "f2": []})
            if math.isfinite(f1):
                per_vowel[v]["f1"].append(f1)
            if math.isfinite(f2):
                per_vowel[v]["f2"].append(f2)
    med = {}
    for v, lists in per_vowel.items():
        med[v] = {}
        for k in ("f1","f2"):
            vals = lists.get(k, [])
            med[v][k] = statistics.median(vals) if vals else None
    return med

class TestProfileVsDiagnostics(unittest.TestCase):
    def test_profile_and_csv_consistency(self):
        if not os.path.exists(PROFILE_PATH):
            self.skipTest("Profile not found")
        profile = load_profile(PROFILE_PATH)
        failures = check_profile(profile)
        self.assertFalse(failures, f"Profile plausibility failures: {failures}")

        if os.path.exists(CSV_PATH):
            mismatches = []
            med = csv_medians_by_vowel(CSV_PATH)
            for v, rec in profile.items():
                pf1 = rec.get("f1"); pf2 = rec.get("f2")
                cf1 = med.get(v, {}).get("f1"); cf2 = med.get(v, {}).get("f2")
                if cf1 is not None and pf1 is not None and abs(pf1 - cf1) > MISMATCH_TOLERANCE_HZ:
                    mismatches.append(f"{v}: profile f1 {pf1:.1f} vs csv median {cf1:.1f}")
                if cf2 is not None and pf2 is not None and abs(pf2 - cf2) > MISMATCH_TOLERANCE_HZ:
                    mismatches.append(f"{v}: profile f2 {pf2:.1f} vs csv median {cf2:.1f}")
            self.assertFalse(mismatches, f"Mismatches vs CSV: {mismatches}")
        else:
            self.skipTest("No report.csv found")

if __name__ == "__main__":
    unittest.main(verbosity=2)