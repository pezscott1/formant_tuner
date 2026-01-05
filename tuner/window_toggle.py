from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt
import numpy as np


class ModeToggleBar(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

        self.btn_analysis = QPushButton("Analysis")
        self.btn_vowelmap = QPushButton("Vowel Map")

        self.btn_analysis.setCheckable(True)
        self.btn_vowelmap.setCheckable(True)
        self.btn_analysis.setChecked(True)

        self.btn_analysis.clicked.connect(self.show_analysis)  # type: ignore
        self.btn_vowelmap.clicked.connect(self.show_vowelmap)  # type: ignore
        self.setFixedHeight(50)
        self.setContentsMargins(0, 0, 0, 0)
        self.btn_vowelmap.setFixedHeight(28)
        self.btn_analysis.setFixedHeight(28)
        self.btn_vowelmap.setStyleSheet("padding: 2px 6px; font-size: 9pt;")
        self.btn_analysis.setStyleSheet("padding: 2px 6px; font-size: 9pt;")

        layout = QHBoxLayout()
        layout.addWidget(self.btn_analysis)
        layout.addWidget(self.btn_vowelmap)
        layout.addStretch()
        self.setLayout(layout)

    def show_analysis(self):
        self.btn_analysis.setChecked(True)
        self.btn_vowelmap.setChecked(False)
        self.stack.setCurrentIndex(0)

    def show_vowelmap(self):
        self.btn_analysis.setChecked(False)
        self.btn_vowelmap.setChecked(True)
        self.stack.setCurrentIndex(1)


class AnalysisView(QWidget):
    def __init__(self, splitter, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)


class VowelMapView(QWidget):
    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.setMinimumSize(300, 300)
        self.analyzer = None
        self.last_valid_f1 = None
        self.last_valid_f2 = None
        self.hold_frames = 10  # number of frames to keep the dot alive
        self.hold_counter = 0
        self.f1_min = None
        self.f1_max = None
        self.f2_min = None
        self.f2_max = None

    VOWEL_COLORS = {
        "i": QColor(0, 200, 255),
        "e": QColor(0, 150, 255),
        "æ": QColor(255, 100, 100),
        "ɑ": QColor(255, 80, 0),
        "ʌ": QColor(255, 180, 0),
        "u": QColor(100, 255, 100),
        "o": QColor(80, 200, 80),
    }

    def update_from_bus(self, bus, analyzer=None):
        self.bus = bus
        if analyzer:
            self.analyzer = analyzer
        self.update()

    def draw_grid(self, painter, w, h):
        if (self.f1_min is None or self.f1_max is None
                or self.f2_min is None or self.f2_max is None):
            return
        painter.setPen(QPen(QColor(60, 60, 60), 1))

        # Vertical gridlines (F2)
        for f2 in [500, 750, 1100, 1600, 2300, 3000]:
            x = int(self.f2_to_x(f2, w))
            painter.drawLine(x, 0, x, h)

        # Horizontal gridlines (F1)
        for f1 in [200, 250, 315, 400, 500, 630, 800, 900]:
            y = int(self.f1_to_y(f1, h))
            painter.drawLine(0, y, w, y)

    def draw_trail(self, painter, w, h, length=40):
        painter.setPen(QPen(QColor(0, 200, 0, 120), 2))

        points = []
        # Take last N frames
        for f1, f2 in list(zip(self.bus.f1, self.bus.f2))[-length:]:
            if f1 is None or f2 is None:
                continue
            x = int(self.f2_to_x(f2, w))
            y = int(self.f1_to_y(f1, h))
            points.append((x, y))

        # Draw polyline
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            painter.drawLine(x1, y1, x2, y2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 35, 35))

        w = self.width()
        h = self.height()

        self.draw_grid(painter, w, h)
        self.draw_targets(painter, w, h)
        self.draw_trail(painter, w, h)

        # Live dot (with hold buffer)
        f1 = self.bus.f1[-1] if self.bus.f1 else None
        f2 = self.bus.f2[-1] if self.bus.f2 else None

        # --- HOLD BUFFER LOGIC GOES HERE ---
        if f1 is not None and f2 is not None:
            # new valid frame
            self.last_valid_f1 = f1
            self.last_valid_f2 = f2
            self.hold_counter = self.hold_frames
        else:
            # fallback to last known good frame
            if self.hold_counter > 0:
                f1 = self.last_valid_f1
                f2 = self.last_valid_f2
                self.hold_counter -= 1
            else:
                f1 = f2 = None

        # --- NOW draw the dot if we have something to draw ---
        if f1 is not None and f2 is not None:
            x = float(self.f2_to_x(f2, w))
            y = float(self.f1_to_y(f1, h))

            # stability ring
            score = self.bus.stability_score[-1] if self.bus.stability_score else None
            conf = self.bus.confidence[-1] if self.bus.confidence else 0.0

            if score is not None:
                if not np.isfinite(score):
                    score = 1.0
                radius = max(10, min(40, int(score * 80)))
                alpha = int(conf * 255)
                painter.setPen(QPen(QColor(255, 255, 255, alpha), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(int(x - radius // 2),
                                    int(y - radius // 2), radius, radius)

            # vowel-colored dot
            vowel = self.bus.vowels[-1] if self.bus.vowels else None
            dot_color = self.VOWEL_COLORS.get(vowel, QColor(80, 255, 80))

            painter.setPen(Qt.NoPen)
            painter.setBrush(dot_color)
            painter.drawEllipse(int(x - 5), int(y - 5), 10, 10)

    def f1_to_y(self, f1, h):
        if f1 is None or self.f1_min is None or self.f1_max is None:
            return -1000  # offscreen sentinel
        y_norm = ((np.log10(f1) - np.log10(self.f1_min)) /
                  (np.log10(self.f1_max) - np.log10(self.f1_min)))
        return h * y_norm

    def f2_to_x(self, f2, w):
        if f2 is None or self.f2_min is None or self.f2_max is None:
            return -1000
        x_norm = ((np.log10(f2) - np.log10(self.f2_min)) /
                  (np.log10(self.f2_max) - np.log10(self.f2_min)))
        return w * (1 - x_norm)

    def draw_targets(self, painter, w, h):
        if not self.analyzer or not hasattr(self.analyzer, "user_formants"):
            return
        if self.analyzer and hasattr(self.analyzer, "user_formants"):
            targets = {}
            for vowel, entry in self.analyzer.user_formants.items():
                f1 = entry.get("f1")
                f2 = entry.get("f2")
                if f1 and f2:
                    targets[vowel] = (f1, f2)

            for vowel, (f1, f2) in targets.items():
                x = float(self.f2_to_x(f2, w))
                y = float(self.f1_to_y(f1, h))

                color = self.VOWEL_COLORS.get(vowel, QColor(180, 180, 180))
                painter.setPen(QPen(color, 2))
                painter.drawEllipse(int(x - 20), int(y - 20), 40, 40)

                # Quadrant-aware label placement
                label_dx = 24 if x < w / 2 else -30
                label_dy = -10 if y < h / 2 else 20
                painter.drawText(int(x + label_dx), int(y + label_dy), vowel)

    def compute_dynamic_ranges(self):
        if not self.analyzer or not hasattr(self.analyzer, "user_formants"):
            return

        f1_vals = []
        f2_vals = []

        for entry in self.analyzer.user_formants.values():
            f1 = entry.get("f1")
            f2 = entry.get("f2")
            if f1:
                f1_vals.append(f1)
            if f2:
                f2_vals.append(f2)

        if f1_vals:
            self.f1_min = min(f1_vals) * 0.9
            self.f1_max = max(f1_vals) * 1.1
        if f2_vals:
            self.f2_min = min(f2_vals) * 0.9
            self.f2_max = max(f2_vals) * 1.1
