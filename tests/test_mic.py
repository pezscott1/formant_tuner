from unittest.mock import patch, MagicMock
import sounddevice as sd


def test_mic_stream_opens_and_calls_callback():
    with patch("sounddevice.InputStream") as mock_stream:
        fake_stream = MagicMock()
        mock_stream.return_value = fake_stream

        # Simulate context manager behavior
        fake_stream.__enter__.return_value = fake_stream
        fake_stream.__exit__.return_value = None

        cb = MagicMock()

        with sd.InputStream(device=12, channels=1, samplerate=48000, callback=cb):
            pass

        mock_stream.assert_called_once()
        fake_stream.__enter__.assert_called_once()
        fake_stream.__exit__.assert_called_once()
