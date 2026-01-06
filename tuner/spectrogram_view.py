from calibration.plotter import safe_spectrogram
import numpy as np
from matplotlib import cm
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QPen, QColor
from PyQt6.QtCore import Qt


class SpectrogramView(QWidget):
    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.setMinimumHeight(200)
        self.setAutoFillBackground(True)

        # EXACT same colormap as calibration
        self.cmap = cm.get_cmap("magma")

        # Fixed analysis parameters
        self.sr = 48000
        self.window_seconds = 2.0
        self.min_hop = 32
        self.max_hop = 512
        self.min_bins = 128
        self.max_bins = 2048
        self.fmin = 50.0
        self.fmax = 4000.0
        self.db_range = 60.0  # dB dynamic range

    def update_from_bus(self, bus):
        self.bus = bus
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Always paint a full background so the widget keeps its full visual size
        painter.fillRect(self.rect(), QColor("black"))

        # ---------------------------------------------------------------------
        # 1) Gather audio
        # ---------------------------------------------------------------------
        chunks = [a for a in getattr(self.bus, "audio", [])
                  if a is not None and len(a) > 0]
        if not chunks:
            return

        audio = np.concatenate(chunks)
        if audio.size < 1024:
            return

        # ---------------------------------------------------------------------
        # 2) Adaptive resolution based on widget size
        # ---------------------------------------------------------------------
        target_h = max(1, self.height())
        target_w = max(1, self.width())

        # Vertical resolution: one bin per pixel (clamped)
        num_bins = int(np.clip(target_h, self.min_bins, self.max_bins))

        # Horizontal resolution: choose hop_length so frames ~= widget width
        desired_frames = max(64, target_w)
        hop_length = int(self.sr * self.window_seconds / desired_frames)
        hop_length = int(np.clip(hop_length, self.min_hop, self.max_hop))

        # ---------------------------------------------------------------------
        # 3) Compute spectrogram (linear frequency)
        # ---------------------------------------------------------------------
        freqs, times, S = safe_spectrogram(
            y=audio,
            sr=self.sr,
            n_fft=1024,
            hop_length=hop_length,
            window_seconds=self.window_seconds,
        )

        if S is None or S.size == 0:
            painter.fillRect(self.rect(), QColor("black"))
            return

        # ---------------------------------------------------------------------
        # 4) Limit to analysis band and remap to log-frequency
        # ---------------------------------------------------------------------
        mask = freqs <= self.fmax
        S = S[mask, :]
        _freqs = freqs[mask]

        if S.size == 0:
            return

        log_freqs = np.logspace(
            np.log10(self.fmin),
            np.log10(self.fmax),
            num_bins,
        )

        S_log = np.zeros((num_bins, S.shape[1]), dtype=float)
        for i in range(S.shape[1]):
            S_log[:, i] = np.interp(log_freqs, _freqs, S[:, i])

        S = S_log
        _freqs = log_freqs

        # ---------------------------------------------------------------------
        # 5) Per-column dB normalization (high contrast, speech-friendly)
        # ---------------------------------------------------------------------
        arr_db = 10.0 * np.log10(S + 1e-12)

        # Per-column max normalization: brightest bin = 0 dB in each frame
        col_max = np.max(arr_db, axis=0, keepdims=True)
        arr_db = arr_db - col_max

        # Clip to fixed dynamic range
        arr_db = np.clip(arr_db, -self.db_range, 0.0)

        # Normalize to [0, 1] for colormap
        norm = (arr_db + self.db_range) / self.db_range

        # ---------------------------------------------------------------------
        # 6) Map to RGB and create QImage
        # ---------------------------------------------------------------------
        rgba = self.cmap(norm)[:, :, :3]  # drop alpha
        img_rgb = (rgba * 255).astype(np.uint8)

        h_img, w_img, _ = img_rgb.shape
        if h_img <= 0 or w_img <= 0:
            return

        qimg = QImage(
            img_rgb.data,
            w_img,
            h_img,
            w_img * 3,
            QImage.Format.Format_RGB888,
        )

        # ---------------------------------------------------------------------
        # 7) Scale to widget using KeepAspectRatioByExpanding (B behavior)
        #    - preserves aspect ratio
        #    - fills both dimensions
        #    - crops overflow
        # ---------------------------------------------------------------------
        scaled = qimg.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        w_scaled = scaled.width()
        h_scaled = scaled.height()

        # Center the expanded image; may be negative if cropping occurs
        x0 = (self.width() - w_scaled) // 2
        y0 = (self.height() - h_scaled) // 2

        # Draw spectrogram with transparency
        painter.setOpacity(0.6)
        painter.drawImage(x0, y0, scaled)
        painter.setOpacity(1.0)

        # ---------------------------------------------------------------------
        # 8) Draw formant overlays aligned to the scaled image
        # ---------------------------------------------------------------------
        self._draw_formants(painter, x0, y0, w_scaled, h_scaled)

    def _draw_formants(self, painter, x0, y0, w, h):
        f1_series = getattr(self.bus, "f1", [])
        f2_series = getattr(self.bus, "f2", [])

        n = len(f1_series)
        if n < 2 or len(f2_series) != n:
            return

        # y: log-frequency mapping 50–4000 Hz → [y0, y0 + h]
        def fy(freq):
            if freq is None or freq <= 0:
                return None
            y_norm = (
                (np.log10(freq) - np.log10(self.fmin)) /
                (np.log10(self.fmax) - np.log10(self.fmin))
            )
            return y0 + int(h * (1.0 - y_norm))

        # x: index → [x0, x0 + w]
        def fx(j):
            return x0 + int(j * w / max(1, n - 1))

        # F1 trajectory
        painter.setPen(QPen(QColor(255, 80, 80), 2))
        for i in range(n - 1):
            f1a = f1_series[i]
            f1b = f1_series[i + 1]
            if f1a is None or f1b is None:
                continue
            y1 = fy(f1a)
            y2 = fy(f1b)
            if y1 is not None and y2 is not None:
                painter.drawLine(fx(i), y1, fx(i + 1), y2)

        # F2 trajectory
        painter.setPen(QPen(QColor(80, 255, 80), 2))
        for i in range(n - 1):
            f2a = f2_series[i]
            f2b = f2_series[i + 1]
            if f2a is None or f2b is None:
                continue
            y1 = fy(f2a)
            y2 = fy(f2b)
            if y1 is not None and y2 is not None:
                painter.drawLine(fx(i), y1, fx(i + 1), y2)

        # Labels at the latest valid F1/F2
        last_f1 = next((v for v in reversed(f1_series) if v is not None), None)
        if last_f1 is not None:
            y_last_f1 = fy(last_f1)
            if y_last_f1 is not None:
                painter.setPen(QPen(QColor(255, 80, 80), 2))
                painter.drawText(x0 + w - 50, y_last_f1, "F1")

        last_f2 = next((v for v in reversed(f2_series) if v is not None), None)
        if last_f2 is not None:
            y_last_f2 = fy(last_f2)
            if y_last_f2 is not None:
                painter.setPen(QPen(QColor(80, 255, 80), 2))
                painter.drawText(x0 + w - 50, y_last_f2, "F2")
