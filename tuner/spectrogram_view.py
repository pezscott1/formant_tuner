from calibration.plotter import safe_spectrogram
import numpy as np
from matplotlib import cm
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QImage, QPen, QColor
from PyQt5.QtCore import Qt


class SpectrogramView(QWidget):
    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.setMinimumHeight(200)
        self.setAutoFillBackground(True)

        # EXACT same colormap as calibration
        self.cmap = cm.get_cmap("magma")

    def update_from_bus(self, bus):
        self.bus = bus
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        if len(self.bus.audio) == 0:
            painter.fillRect(self.rect(), Qt.black)
            return

        audio = np.concatenate([a for a in self.bus.audio if a is not None])
        if audio.size < 1024:
            painter.fillRect(self.rect(), Qt.black)
            return

        # Calibration-style spectrogram
        freqs, times, S = safe_spectrogram(
            y=audio,
            sr=48000,
            n_fft=1024,
            hop_length=256,
            window_seconds=1.0,
        )

        if S is None or S.size == 0:
            painter.fillRect(self.rect(), Qt.black)
            return

        # Mask to 0–4 kHz (same as calibration)
        mask = freqs <= 4000
        S = S[mask, :]
        _freqs = freqs[mask]
        # --- NEW: log-frequency resampling for spectrogram image ---
        fmin, fmax = 50, 4000
        num_bins = 256  # or 512 for smoother vertical resolution
        log_freqs = np.logspace(np.log10(fmin), np.log10(fmax), num_bins)

        S_log = np.zeros((num_bins, S.shape[1]))
        for i in range(S.shape[1]):
            S_log[:, i] = np.interp(log_freqs, _freqs, S[:, i])

        S = S_log
        _freqs = log_freqs

        # dB scaling (identical to calibration)
        arr_db = 10 * np.log10(S + 1e-12)
        arr_db_max = np.max(arr_db)
        arr_db = np.clip(arr_db, arr_db_max - 60, arr_db_max)

        # Normalize to 0–1 for colormap
        norm = (arr_db - (arr_db_max - 60)) / 60

        # Apply magma colormap (identical to calibration)
        rgba = self.cmap(norm)[:, :, :3]  # drop alpha
        img_rgb = (rgba * 255).astype(np.uint8)

        h, w, _ = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)

        painter.drawImage(self.rect(), qimg)

        self._draw_formants(painter, w, h)

    def _draw_formants(self, painter, _w, h):
        n = len(self.bus.f1)
        if n < 2:
            return

        def fy(freq):
            if freq is None or freq <= 0:
                return None
            y_norm = (np.log10(freq) - np.log10(50)) / (np.log10(4000) - np.log10(50))
            return int(h * (1.0 - y_norm))

        def fx(j):
            return int(j * self.width() / n)

        # F1 (red)
        painter.setPen(QPen(QColor(255, 80, 80), 2))
        for i in range(n - 1):
            if self.bus.f1[i] is None or self.bus.f1[i + 1] is None:
                continue
            y1 = fy(self.bus.f1[i])
            y2 = fy(self.bus.f1[i + 1])
            if y1 is not None and y2 is not None:
                painter.drawLine(fx(i), y1, fx(i + 1), y2)

        # F2 (green)
        painter.setPen(QPen(QColor(80, 255, 80), 2))
        for i in range(n - 1):
            if self.bus.f2[i] is None or self.bus.f2[i + 1] is None:
                continue
            y1 = fy(self.bus.f2[i])
            y2 = fy(self.bus.f2[i + 1])
            if y1 is not None and y2 is not None:
                painter.drawLine(fx(i), y1, fx(i + 1), y2)

        # Labels
        last_f1 = next((v for v in reversed(self.bus.f1) if v is not None), None)
        if last_f1 is not None:
            painter.setPen(QPen(QColor(255, 80, 80), 2))
            painter.drawText(self.width() - 50, fy(last_f1), "F1")

        last_f2 = next((v for v in reversed(self.bus.f2) if v is not None), None)
        if last_f2 is not None:
            painter.setPen(QPen(QColor(80, 255, 80), 2))
            painter.drawText(self.width() - 50, fy(last_f2), "F2")
