import pytest
from unittest.mock import MagicMock
from formant_tuner import ProfileManager  # adjust import path if needed


def test_delete_profile_missing_file(tmp_path):
    pm = ProfileManager(tmp_path, analyzer=None)
    pm.delete_profile("nonexistent")  # should not raise


def test_apply_profile_missing_file(tmp_path):
    analyzer = MagicMock()
    analyzer.load_profile.side_effect = FileNotFoundError
    pm = ProfileManager(tmp_path, analyzer=analyzer)
    with pytest.raises(FileNotFoundError):
        pm.apply_profile("ghost")
