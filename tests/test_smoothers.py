from voice_analysis import MedianSmoother, FormantSmoother


def test_median_smoother_none_behavior():
    s = MedianSmoother(size=3)
    assert s.update(None, None, None) == (None, None, None)
    assert s.update(300, 1200, 0) != (None, None, None)


def test_formant_smoother_median():
    s = FormantSmoother(size=3)
    f1, f2 = s.update(300, 1200)
    assert f1 == 300 and f2 == 1200
    s.update(320, 1180)
    s.update(310, 1190)
    f1m, f2m = s.update(None, None)
    assert 305 <= f1m <= 320
    assert 1180 <= f2m <= 1200
