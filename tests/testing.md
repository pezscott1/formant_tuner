# Testing & Diagnostics Framework

This project uses a layered testing approach to ensure that vowel calibration, diagnostics, and profile persistence are reliable and pedagogically meaningful. There are three main categories of tests:

---

## 1. Unit Tests (`tests/test_formant.py`)

**Purpose:** Verify correctness of core functions and utilities.

**Coverage:**
- **Formant estimation** (`estimate_formants_lpc`) on synthetic signals.
- **Vowel guessing** (`robust_guess`) with clear, missing, uncertain, and tie cases.
- **Smoothing** (`MedianSmoother`, `StableTracker`) to reject implausible values and stabilize outputs.
- **Analyzer rendering** (status text, vowel chart, spectrum, diagnostics).
- **Scoring functions** (`live_score_formants`, `resonance_tuning_score`, `overall_rating`).

**Outcome:** Confirms that the building blocks of the calibration and tuner pipeline behave correctly in isolation.

---

## 2. Integration Tests (`tests/test_profile_vs_diagnostics.py`)

**Purpose:** Validate that saved calibration profiles are consistent with diagnostic CSV data.

**Coverage:**
- Loads `profiles/user1_profile.json`.
- Checks plausibility of each vowel record (finite values, ranges, F1 < F2, minimum separation).
- Loads `report.csv` (if present) and computes median F1/F2 per vowel.
- Compares profile values against CSV medians within a tolerance (±100 Hz).

**Outcome:** Ensures that calibration results match diagnostic measurements and that profiles are internally consistent.

---

## 3. Persistence & Schema Tests (`tests/test_saved_profiles.py`)

**Purpose:** Confirm that saved profile JSON files exist, have the correct schema, and contain plausible values.

**Coverage:**
- Asserts that `profiles/user1_profile.json` exists.
- Validates schema: each vowel record must contain `f1`, `f2`, and `retries`.
- Enforces ordering (`f1 ≤ f2`).
- Uses `is_plausible_formants` to check plausibility of formant pairs.

**Outcome:** Guarantees that persisted profiles are structurally valid and pedagogically meaningful.

---

## Test Strategy Summary

- **Unit layer**: `test_formant.py` — correctness of algorithms and rendering.
- **Integration layer**: `test_profile_vs_diagnostics.py` — consistency between calibration profiles and diagnostic CSV data.
- **Persistence layer**: `test_saved_profiles.py` — schema and plausibility of saved profiles.

Together, these three files replace older redundant tests (e.g. `test_stable.py`) and provide full coverage of:
- Signal analysis
- Vowel classification
- Smoothing/stability
- Profile saving/loading
- Diagnostic consistency

---

## Running Tests

You can run all tests with:

```bash
pytest -v