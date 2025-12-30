def test_te_noise_collapse_branch():
    import numpy as np
    from analysis.true_envelope import estimate_formants_te

    noise = np.random.randn(2048)
    res = estimate_formants_te(noise, sr=48000)

    assert res.f1 is None
    assert res.f2 is None
    assert res.debug["reason"] == "std_len_noise_collapse"


def test_te_no_band_branch():
    import numpy as np
    from analysis.true_envelope import estimate_formants_te

    x = np.ones(200)  # arbitrary
    res = estimate_formants_te(x, sr=100)  # 100 Hz sample rate → rfftfreq < 50 Hz

    assert res.f1 is None
    assert res.debug["reason"] == "no_band"


def test_te_no_peaks_branch():
    import numpy as np
    from analysis.true_envelope import estimate_formants_te

    # Very short signal → vowel band empty or flat → no peaks
    x = np.zeros(32)
    res = estimate_formants_te(x, sr=48000)

    assert res.f1 == 0.0
    assert res.f2 == 0.0
    assert res.debug["reason"] == "no_peaks"


def test_te_extract_formants_import_fallback(monkeypatch):
    import numpy as np
    from analysis import true_envelope

    # Force import failure
    monkeypatch.setattr(true_envelope, "analysis", None, raising=False)

    # Simple signal with one peak
    t = np.linspace(0, 0.02, 1000, endpoint=False)
    x = np.sin(2*np.pi*500*t)

    res = true_envelope.estimate_formants_te(x, sr=48000)

    assert res.f1 is not None
    assert res.f2 is not None


def test_te_fallback_formant_assignment(monkeypatch):
    import numpy as np
    from analysis import true_envelope

    def fake_extract(_):
        return None, None, None

    monkeypatch.setattr(true_envelope, "_extract_formants", fake_extract, raising=False)

    t = np.linspace(0, 0.02, 1000, endpoint=False)
    x = np.sin(2*np.pi*500*t) + 0.5*np.sin(2*np.pi*1500*t)

    res = true_envelope.estimate_formants_te(x, sr=48000)

    assert res.f1 < res.f2  # sorted peaks
