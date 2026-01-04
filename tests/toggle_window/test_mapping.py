from tuner.window_toggle import VowelMapView


def test_mapping_functions(qtbot):
    vm = VowelMapView(bus=None)
    vm.f1_min, vm.f1_max = 200, 800
    vm.f2_min, vm.f2_max = 500, 3000

    y_low = vm.f1_to_y(200, 1000)
    y_high = vm.f1_to_y(800, 1000)
    assert y_low < y_high

    x_low = vm.f2_to_x(3000, 1000)
    x_high = vm.f2_to_x(500, 1000)
    assert x_low < x_high

    # monotonicity
    assert vm.f1_to_y(300, 1000) < vm.f1_to_y(600, 1000)
    assert vm.f2_to_x(2500, 1000) < vm.f2_to_x(1000, 1000)
