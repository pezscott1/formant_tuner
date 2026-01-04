from tuner.window_toggle import VowelMapView


class FakeAnalyzer:
    user_formants = {
        "i": {"f1": 300, "f2": 2500},
        "e": {"f1": 400, "f2": 2000},
        "æ": {"f1": 700, "f2": 1700},
    }


def test_compute_dynamic_ranges(qtbot):
    vm = VowelMapView(bus=None)
    vm.analyzer = FakeAnalyzer()
    vm.compute_dynamic_ranges()

    assert vm.f1_min < 300
    assert vm.f1_max > 700
    assert vm.f2_min < 1700
    assert vm.f2_max > 2500

    # log-space padding should be nonzero
    assert vm.f1_max - vm.f1_min > 0
    assert vm.f2_max - vm.f2_min > 0
