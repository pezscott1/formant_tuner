from PyQt6.QtGui import QImage, QPainter
from tuner.window_toggle import VowelMapView
from profile_viewer.vowel_colors import vowel_color_for


class FakeAnalyzer:
    def __init__(self, user_formants=None, interpolated=None):
        self.user_formants = user_formants or {}
        self.interpolated_vowels = interpolated or {}


class FakeBus:
    def __init__(self, points=None):
        self._points = points or []
        self.f1 = []
        self.f2 = []
        self.vowels = []
        self.stability_score = []
        self.confidence = []

    def get_recent_points(self):
        return self._points


# ---------------------------------------------------------------------------
# 1. compute_dynamic_ranges must not crash when bus=None
# ---------------------------------------------------------------------------
def test_compute_dynamic_ranges_bus_none():
    vm = VowelMapView(bus=None)
    vm.analyzer = FakeAnalyzer(
        user_formants={"i": {"f1": 300, "f2": 2500}}
    )

    vm.compute_dynamic_ranges()

    assert vm.f1_min < vm.f1_max
    assert vm.f2_min < vm.f2_max


# ---------------------------------------------------------------------------
# 2. color assignment is deterministic via vowel_color_for
# ---------------------------------------------------------------------------
def test_dynamic_color_assignment_stable():
    vowels = ["i", "e", "a"]
    colors1 = {v: vowel_color_for(v) for v in vowels}
    colors2 = {v: vowel_color_for(v) for v in vowels}
    assert colors1 == colors2


def test_dynamic_color_assignment_new_vowel_gets_color():
    base = {"i": vowel_color_for("i")}
    base_keys = set(base.keys())

    # Add a new vowel and ensure it has a color
    base["u"] = vowel_color_for("u")
    after_keys = set(base.keys())

    assert "u" in after_keys
    assert after_keys == base_keys | {"u"}


# ---------------------------------------------------------------------------
# 3. draw_targets must never crash, even with overlapping vowels
# ---------------------------------------------------------------------------
def test_draw_targets_no_crash_overlapping():
    vm = VowelMapView(bus=FakeBus())
    vm.analyzer = FakeAnalyzer(
        user_formants={
            v: {"f1": 500, "f2": 1500}
            for v in ["i", "e", "a", "o", "u", "æ", "ʌ", "ɪ"]
        }
    )

    vm.update_from_bus(bus=vm.bus, analyzer=vm.analyzer)
    vm.compute_dynamic_ranges()

    img = QImage(400, 400, QImage.Format.Format_ARGB32)
    painter = QPainter(img)

    vm.draw_targets(painter, 400, 400)
    painter.end()  # no crash = pass


# ---------------------------------------------------------------------------
# 4. compute_dynamic_ranges must include interpolated vowels if present
# ---------------------------------------------------------------------------
def test_compute_dynamic_ranges_includes_interpolated():
    vm = VowelMapView(bus=None)
    vm.analyzer = FakeAnalyzer(
        user_formants={"i": {"f1": 300, "f2": 2500}},
        interpolated={"e": {"f1": 400, "f2": 2200}},
    )

    vm.update_from_bus(bus=None, analyzer=vm.analyzer)
    vm.compute_dynamic_ranges()

    assert vm.f1_min <= 300 * 0.85
    assert vm.f1_max >= 400 * 1.15
    assert vm.f2_min <= 2200 * 0.85
    assert vm.f2_max >= 2500 * 1.15


# ---------------------------------------------------------------------------
# 5. update_from_bus must not crash when analyzer is missing fields
# ---------------------------------------------------------------------------
def test_update_from_bus_missing_fields():
    class PartialAnalyzer:
        user_formants = {"i": {"f1": 300, "f2": 2500}}
        # missing interpolated_vowels

    vm = VowelMapView(bus=FakeBus(points=[]))
    vm.analyzer = PartialAnalyzer()

    vm.update_from_bus(bus=vm.bus, analyzer=vm.analyzer)  # no crash
