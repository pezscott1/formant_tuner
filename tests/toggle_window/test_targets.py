from PyQt5.QtGui import QPainter, QImage
from tuner.window_toggle import VowelMapView


class FakeAnalyzer:
    user_formants = {
        "i": {"f1": 300, "f2": 2500},
        "e": {"f1": 400, "f2": 2000},
    }


def test_draw_targets_no_crash(qtbot):
    vm = VowelMapView(bus=None)
    vm.analyzer = FakeAnalyzer()
    vm.compute_dynamic_ranges()

    img = QImage(400, 400, QImage.Format_ARGB32)
    painter = QPainter(img)

    # Should not crash
    vm.draw_targets(painter, 400, 400)
    painter.end()
