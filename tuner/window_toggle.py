# tuner/window_toggle.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QCheckBox)
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QRect
import numpy as np
from profile_viewer.vowel_colors import vowel_color_for


class ModeToggleBar(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.setStyleSheet("background-color: #f0f0f0;")
        self.btn_analysis = QPushButton("Analysis")
        self.btn_vowelmap = QPushButton("Vowel Map")
        self.btn_analysis.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.btn_vowelmap.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.btn_analysis.setCheckable(True)
        self.btn_vowelmap.setCheckable(True)
        self.btn_analysis.setChecked(True)

        self.btn_analysis.clicked.connect(self.show_analysis)  # type: ignore
        self.btn_vowelmap.clicked.connect(self.show_vowelmap)  # type: ignore

        self.setFixedHeight(50)
        self.setContentsMargins(4, 4, 4, 0)
        self.btn_vowelmap.setFixedHeight(28)
        self.btn_analysis.setFixedHeight(28)
        self.btn_analysis.setStyleSheet("""
        QPushButton {
            background-color: #f7c6c6;      /* inactive = red */
            color: black;
            border: 1px solid #a66;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 9pt;
        }
        QPushButton:hover {
            background-color: #f2b3b3;      /* deeper red */
        }
        QPushButton:checked {
            background-color: #c6f7c6;      /* active = green */
            border: 1px solid #4caf50;
        }
        QPushButton:checked:hover {
            background-color: #b3f2b3;      /* deeper green */
        }
        """)

        self.btn_vowelmap.setStyleSheet("""
        QPushButton {
            background-color: #f7c6c6;      /* inactive = red */
            color: black;
            border: 1px solid #a66;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 9pt;
        }
        QPushButton:hover {
            background-color: #f2b3b3;      /* deeper red */
        }
        QPushButton:checked {
            background-color: #c6f7c6;      /* active = green */
            border: 1px solid #4caf50;
        }
        QPushButton:checked:hover {
            background-color: #b3f2b3;      /* deeper green */
        }
        """)

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
        self.setStyleSheet("background-color: #2b2b2b;")


class VowelMapView(QWidget):
    def __init__(self, bus, parent=None):
        super().__init__(parent)

        self.bus = bus
        self.setMinimumSize(300, 300)
        self.analyzer = None
        self.calibrated_vowels = set()
        self.interpolated_vowels = set()

        # Hold-buffer for dot persistence
        self.last_valid_f1 = None
        self.last_valid_f2 = None
        self.hold_frames = 10
        self.hold_counter = 0

        # Dynamic ranges
        self.f1_min = None
        self.f1_max = None
        self.f2_min = None
        self.f2_max = None

        self.show_interp = QCheckBox("Show interpolated vowels")
        self.show_interp.setStyleSheet("""
            color: black;
            QCheckBox::indicator {
                border: 1px solid black;
                background: white;
            }
            QCheckBox::indicator:checked {
                background: #4caf50;
                border: 1px solid #4caf50;
            }
        """)
        self.show_interp.setChecked(True)
        self.show_interp.stateChanged.connect(self.update)  # type: ignore
        self.canvas_area = QWidget()
        self.canvas_area.setMinimumSize(300, 300)
        self.canvas_area.paintEvent = self.paint_canvas  # redirect painter

        legend = QLabel(" ○ Calibrated = colors  ○ Interpolated = grey ")
        legend.setStyleSheet("color: black; font-size: 9pt; background: transparent;")

        legend_bar = QHBoxLayout()
        legend_bar.addWidget(legend)
        legend_bar.addWidget(self.show_interp)
        legend_bar.addStretch()
        legend_bar.setContentsMargins(8, 4, 8, 4)

        # Wrap legend bar in its own widget so it has its own background
        legend_bar_widget = QWidget()
        legend_bar_widget.setStyleSheet("background-color: #f0f0f0;")
        legend_bar_widget.setLayout(legend_bar)

        title = QLabel("Vowel Map")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 16pt;
            font-weight: bold;
            color: black;
            padding: 10px 6px 14px 6px;   /* extra bottom padding */
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.canvas_area, stretch=10)  # painter gets most space
        layout.addWidget(legend_bar_widget)  # legend now BELOW canvas
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def update_from_bus(self, bus, analyzer=None):
        self.bus = bus
        # Update analyzer if needed
        if analyzer and analyzer is not self.analyzer:
            self.analyzer = analyzer
        self.compute_dynamic_ranges()
        self.canvas_area.update()

    def set_vowel_status(self, calibrated, interpolated):
        self.calibrated_vowels = set(calibrated)
        self.interpolated_vowels = set(interpolated)

    def update_hold_buffer(self):
        f1 = self.bus.f1[-1] if self.bus.f1 else None
        f2 = self.bus.f2[-1] if self.bus.f2 else None

        if f1 is not None and f2 is not None:
            self.last_valid_f1 = f1
            self.last_valid_f2 = f2
            self.hold_counter = self.hold_frames
        else:
            if self.hold_counter > 0:
                self.hold_counter -= 1

    def paintEvent(self, event):
        # Run hold-buffer logic immediately for tests
        self.update_hold_buffer()
        # Trigger a real repaint of the canvas
        self.canvas_area.update()

    # ------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------

    def draw_grid(self, painter, w, h):
        if (self.f1_min is None or self.f1_max is None
                or self.f2_min is None or self.f2_max is None):
            return

        painter.setPen(QPen(QColor(120, 120, 120), 1))

        # Vertical gridlines (F2)
        for f2 in [500, 750, 1100, 1600, 2300, 3000]:
            x = int(self.f2_to_x(f2, w))
            painter.drawLine(x, 0, x, h)

        # Horizontal gridlines (F1)
        for f1 in [200, 250, 315, 400, 500, 630, 800, 900]:
            y = int(self.f1_to_y(f1, h))
            painter.drawLine(0, y, w, y)

    def draw_trail(self, painter, w, h, length=40):
        points = []
        for f1, f2 in list(zip(self.bus.f1, self.bus.f2))[-length:]:
            if f1 is None or f2 is None:
                continue
            x = int(self.f2_to_x(f2, w))
            y = int(self.f1_to_y(f1, h))
            points.append((x, y))

        for i in range(1, len(points)):
            alpha = int(255 * (i / len(points)))
            painter.setPen(QPen(QColor(0, 200, 0, alpha), 2))
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            painter.drawLine(x1, y1, x2, y2)

    def draw_targets(self, painter, w, h):
        if not self.analyzer or not hasattr(self.analyzer, "user_formants"):
            return
        occupied = []
        offsets = [
            (0, -20), (20, -20), (20, 0), (20, 20),
            (0, 20), (-20, 20), (-20, 0), (-20, -20)
        ]
        for vowel, entry in self.analyzer.user_formants.items():
            f1 = entry.get("f1")
            f2 = entry.get("f2")
            if not f1 or not f2:
                continue

            x = float(self.f2_to_x(f2, w))
            y = float(self.f1_to_y(f1, h))

            # -----------------------------
            # Compute label placement FIRST
            # -----------------------------
            base_x = x + (24 if x < w / 2 else -30)
            base_y = y + (-10 if y < h / 2 else 20)
            label_rect = QRect(int(base_x), int(base_y - 12 - 6), 30, 20)

            # Apply offsets to avoid collisions
            for dx, dy in offsets:
                test_rect = label_rect.translated(dx, dy)
                if not any(test_rect.intersects(r) for r in occupied):
                    label_rect = test_rect
                    break
            # -----------------------------
            # Draw calibrated vs interpolated
            # -----------------------------
            if vowel in self.calibrated_vowels:
                color = QColor(vowel_color_for(vowel))
                painter.setPen(QPen(color, 2))
                painter.drawEllipse(int(x - 20), int(y - 20), 40, 40)

                painter.setPen(QPen(color, 1))
                painter.drawText(label_rect.left(), label_rect.bottom(), vowel)

            elif vowel in self.interpolated_vowels and self.show_interp.isChecked():
                painter.setPen(QPen(QColor("gray"), 2))
                painter.drawEllipse(int(x - 20), int(y - 20), 40, 40)

                painter.setPen(QPen(QColor("gray"), 1))
                painter.drawText(label_rect.left(), label_rect.bottom(), vowel)
            # Mark this label rect as occupied
            occupied.append(label_rect)

    # ------------------------------------------------------------
    # Main paint event
    # ------------------------------------------------------------

    def paint_canvas(self, event):

        painter = QPainter(self.canvas_area)
        rect = self.canvas_area.rect()

        painter.fillRect(rect, QColor(35, 35, 35))
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        painter.drawRect(rect.adjusted(1, 1, -2, -2))

        w = self.canvas_area.width()
        h = self.canvas_area.height()

        self.draw_grid(painter, w, h)
        self.draw_targets(painter, w, h)
        self.draw_trail(painter, w, h)

        # Update hold buffer first
        self.update_hold_buffer()

        # Retrieve the current (possibly held) values
        f1 = self.last_valid_f1
        f2 = self.last_valid_f2

        if f1 is not None and f2 is not None:
            # Smoothing
            alpha = 0.25
            f1 = alpha * f1 + (1 - alpha) * self.last_valid_f1
            f2 = alpha * f2 + (1 - alpha) * self.last_valid_f2

            x = float(self.f2_to_x(f2, w))
            y = float(self.f1_to_y(f1, h))

            score = self.bus.stability_score[-1] if self.bus.stability_score else None
            conf = self.bus.confidence[-1] if self.bus.confidence else 0.0

            if score is not None:
                if not np.isfinite(score):
                    score = 1.0
                radius = max(10, min(40, int(score * 80)))
                alpha_ring = int(conf * 255)
                painter.setPen(QPen(QColor(255, 255, 255, alpha_ring), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(int(x - radius // 2),
                                    int(y - radius // 2), radius, radius)

            vowel = self.bus.vowels[-1] if self.bus.vowels else None
            dot_color = QColor(vowel_color_for(vowel)) if vowel else QColor(80, 255, 80)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(dot_color)
            painter.drawEllipse(int(x - 5), int(y - 5), 10, 10)

    # ------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------

    def f1_to_y(self, f1, h):
        if f1 is None or self.f1_min is None or self.f1_max is None:
            return -1000
        y_norm = ((np.log10(f1) - np.log10(self.f1_min)) /
                  (np.log10(self.f1_max) - np.log10(self.f1_min)))
        return h * y_norm

    def f2_to_x(self, f2, w):
        if f2 is None or self.f2_min is None or self.f2_max is None:
            return -1000
        x_norm = ((np.log10(f2) - np.log10(self.f2_min)) /
                  (np.log10(self.f2_max) - np.log10(self.f2_min)))
        return w * (1 - x_norm)

    # ------------------------------------------------------------
    # Dynamic ranges
    # ------------------------------------------------------------

    def compute_dynamic_ranges(self):
        f1_vals = []
        f2_vals = []

        # Include calibrated vowels
        for v, entry in self.analyzer.user_formants.items():
            f1 = entry.get("f1")
            f2 = entry.get("f2")
            if f1:
                f1_vals.append(f1)
            if f2:
                f2_vals.append(f2)

        # Include interpolated vowels
        interp = None
        if hasattr(self.analyzer, "interpolated_vowels"):
            interp = self.analyzer.interpolated_vowels
        elif hasattr(self.analyzer, "interpolated"):
            interp = self.analyzer.interpolated

        if interp:
            for v in interp.values():
                f1_vals.append(v.get("f1"))
                f2_vals.append(v.get("f2"))

        # Include recent bus points
        if self.bus is not None and hasattr(self.bus, "get_recent_points"):
            for pt in self.bus.get_recent_points():
                f1 = pt.get("f1")
                f2 = pt.get("f2")
                if f1:
                    f1_vals.append(f1)
                if f2:
                    f2_vals.append(f2)

        # Compute axis limits
        if f1_vals:
            self.f1_min = min(f1_vals) * 0.85
            self.f1_max = max(f1_vals) * 1.15
        else:
            self.f1_min = 200
            self.f1_max = 1000

        if f2_vals:
            self.f2_min = min(f2_vals) * 0.85
            self.f2_max = max(f2_vals) * 1.15
        else:
            self.f2_min = 400
            self.f2_max = 3000
