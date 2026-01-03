from unittest.mock import MagicMock
from tuner.controller import Tuner


# ---------------------------------------------------------
# Dummy objects aligned with the NEW Tuner architecture
# ---------------------------------------------------------
class DummyEngine:
    def __init__(self):
        self.voice_type = "bass"
        self.user_formants = None
        self._raw = None
        self.vowel_hint = None

    def get_latest_raw(self):
        return self._raw


class DummyProfileManager:
    def __init__(self):
        self.deleted = []
        self.applied = []
        self.loaded = []

    def list_profiles(self):
        return ["alpha", "beta"]

    def apply_profile(self, base):
        self.applied.append(base)
        return base

    def load_profile_json(self, base):
        self.loaded.append(base)
        return {
            "voice_type": "bass",
            "a": {"f1": 500, "f2": 1500, "f0": 100}
        }

    def extract_formants(self, raw):
        # MUST return dict of dicts for new classifier
        return {"a": {"f1": 500, "f2": 1500, "f0": 100}}

    def delete_profile(self, base):
        self.deleted.append(base)


class DummyLiveAnalyzer:
    """
    Must expose get_latest_processed() for the new Tuner API.
    """
    def __init__(self):
        self.started = False
        self.stopped = False
        self._processed = None

    def start_worker(self):
        self.started = True

    def stop_worker(self):
        self.stopped = True

    def reset(self):
        pass

    def get_latest_processed(self):
        return self._processed


def make_tuner():
    engine = DummyEngine()
    pm = DummyProfileManager()
    la = DummyLiveAnalyzer()
    t = Tuner(engine=engine)
    t.profile_manager = pm
    t.live_analyzer = la
    return t, engine, pm, la


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------
def test_list_profiles():
    t, _, pm, _ = make_tuner()
    assert t.list_profiles() == ["alpha", "beta"]


def test_load_profile_sets_user_formants():
    t, engine, pm, _ = make_tuner()
    t.load_profile("alpha")

    assert "a" in t.active_profile
    vals = t.active_profile["a"]
    assert vals["f1"] == 500
    assert vals["f2"] == 1500
    assert vals["f0"] == 100


def test_delete_profile():
    t, _, pm, _ = make_tuner()
    t.delete_profile("alpha")
    assert pm.deleted == ["alpha"]


def test_update_tolerance():
    t, _, _, _ = make_tuner()
    assert t.update_tolerance("80") == 80
    assert t.update_tolerance("bad") == 80
    assert t.update_tolerance("-10") == 80


def test_start_mic_success():
    t, _, _, la = make_tuner()
    fake_stream = MagicMock()
    ok = t.start_mic(lambda: fake_stream)
    assert ok
    fake_stream.start.assert_called_once()
    assert la.started


def test_start_mic_failure():
    t, _, _, la = make_tuner()
    ok = t.start_mic(lambda: (_ for _ in ()).throw(Exception("boom")))
    assert not ok
    assert not la.started


def test_stop_mic():
    t, _, _, la = make_tuner()
    fake_stream = MagicMock()
    t.stream = fake_stream
    ok = t.stop_mic()
    assert ok
    fake_stream.stop.assert_called_once()
    assert la.stopped


def test_poll_latest_processed():
    t, engine, _, la = make_tuner()

    # Engine raw frame
    engine._raw = {"segment": [1, 2, 3]}

    # Analyzer processed frame
    la._processed = {"f0": 100}

    out = t.poll_latest_processed()
    assert out["f0"] == 100
