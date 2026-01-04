from tuner.window_toggle import VowelMapView


class FakeBus:
    def __init__(self):
        self.f1 = []
        self.f2 = []
        self.vowels = []
        self.stability_score = []
        self.confidence = []


def test_hold_buffer(qtbot):
    bus = FakeBus()
    vm = VowelMapView(bus)

    # First frame: valid
    bus.f1.append(300)
    bus.f2.append(2000)
    bus.vowels.append("i")
    bus.stability_score.append(0.5)
    bus.confidence.append(1.0)

    vm.paintEvent(None)
    assert vm.last_valid_f1 == 300
    assert vm.hold_counter == vm.hold_frames

    # Next frame: missing data
    bus.f1.append(None)
    bus.f2.append(None)
    bus.vowels.append(None)
    bus.stability_score.append(None)
    bus.confidence.append(0)

    vm.paintEvent(None)
    assert vm.last_valid_f1 == 300
    assert vm.hold_counter == vm.hold_frames - 1
