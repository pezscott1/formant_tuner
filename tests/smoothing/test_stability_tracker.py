# tests/smoothing/test_stability_tracker.py
from analysis.smoothing import FormantStabilityTracker


def test_stability_tracker_allows_missing_f3():
    st = FormantStabilityTracker(window_size=6, min_full_frames=3)

    # Feed stable F1/F2, missing F3
    frames = [
        (500, 1500, None),
        (505, 1490, None),
        (495, 1510, None),
        (500, 1505, None),
    ]

    stable = False
    score = float("inf")

    for f1, f2, f3 in frames:
        stable, score = st.update(f1, f2, f3)

    assert stable is True
    assert score < 1e5  # below threshold
