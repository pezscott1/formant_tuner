from tuner.window_toggle import VowelMapView


def test_no_crash_empty_bus(qtbot):
    class EmptyBus:
        f1 = []
        f2 = []
        vowels = []
        stability_score = []
        confidence = []

    vm = VowelMapView(bus=EmptyBus())
    vm.paintEvent(None)  # Should not crash


def test_no_crash_no_analyzer(qtbot):
    class FakeBus:
        f1 = [300]
        f2 = [2000]
        vowels = ["i"]
        stability_score = [0.5]
        confidence = [1.0]

    vm = VowelMapView(bus=FakeBus())
    vm.paintEvent(None)  # Should not crash
