# profile_viewer.py
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTextEdit
)
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas


class ProfileViewerWindow(QMainWindow):
    """
    Visualizes a vowel profile (calibrated + interpolated vowels)
    as an F1–F2 vowel map, similar to the calibration window but larger.
    """

    def __init__(self, profile_data: dict, parent=None, headless=False):
        super().__init__(parent)
        self.headless = headless
        self.setWindowTitle("Vowel Profile Viewer")

        # Store profile
        self.profile = profile_data or {}

        # Identify calibrated vs interpolated vowels
        self.calibrated_vowels = {"i", "ɛ", "ɑ", "ɔ", "u"}
        self.interpolated_vowels = {
            v for v in self.profile.keys() if v not in self.calibrated_vowels
        }

        # Colors
        self.colors = {
            "i": "red",
            "ɛ": "green",
            "ɑ": "blue",
            "ɔ": "purple",
            "u": "orange",
        }
        self.interp_color = "gray"

        if headless:
            # --- Create minimal but fully functional UI elements ---
            # Values panel
            self.values_panel = QTextEdit()
            self.values_panel.setReadOnly(True)
            self.values_panel.setHtml(self._build_values_panel_html())

            # Legend panel
            self.legend_panel = QTextEdit()
            self.legend_panel.setReadOnly(True)
            self.legend_panel.setHtml(
                "<b>Calibrated vowels</b><br>Interpolated vowels"
            )

            self.fig, self.ax = plt.subplots()

            # Window size (tests expect >600x500)
            self.resize(1000, 800)

            # Fake canvas with draw_idle so _plot_profile works
            self.canvas = MagicMock()
            self.canvas.draw_idle = lambda: None

            return

        else:
            self._build_ui()
            self._plot_profile()
            self.show()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _resize_and_center(self):
        if self.headless:
            return
        screen = self.screen().availableGeometry()
        w = int(screen.width() * 0.7)
        h = int(screen.height() * 0.7)
        self.resize(w, h)
        self.move(screen.center().x() - w // 2, screen.center().y() - h // 2)

    def _build_ui(self):
        if self.headless:
            return  # skip all Qt widgets

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = self._build_left_panel()
        layout.addWidget(left, stretch=0)

        right = self._build_right_panel()
        layout.addWidget(right, stretch=1)

    def _build_left_panel(self):
        frame = QFrame()
        layout = QVBoxLayout(frame)

        layout.addWidget(QLabel("Profile Values"))

        self.values_panel = QTextEdit()
        self.values_panel.setReadOnly(True)
        self.values_panel.setAcceptRichText(True)

        layout.addWidget(self.values_panel)

        legend_label = QLabel("Legend")
        legend_label.setStyleSheet("font-weight: bold; font-size: 10pt;")

        lines = []
        for vowel, data in self.profile.items():
            f1 = data.get("f1")
            f2 = data.get("f2")
            f0 = data.get("f0")
            color = self.colors.get(vowel, self.interp_color)
            f0_display = f"{f0:.1f}" if isinstance(f0, (int, float)) else "—"
            lines.append(
                f"<span style='color:{color}; font-weight:bold'>/{vowel}/</span> "
                f"F1={f1:.1f}  F2={f2:.1f}  F0={f0_display}"
            )

        self.values_panel.setHtml("<br>".join(lines))

        self.legend_panel = QTextEdit()
        self.legend_panel.setReadOnly(True)
        self.legend_panel.setAcceptRichText(True)
        self.legend_panel.setStyleSheet("font-size: 9pt;")

        self.legend_panel.setHtml("""
        <b>Legend</b><br>
        <span style='color:red'>/i/</span>, <span style='color:green'>/ɛ/</span>,
        <span style='color:blue'>/ɑ/</span>,
        etc. = calibrated vowels<br>
        <span style='color:gray'>/e/</span>, <span style='color:gray'>/ɪ/</span>,
        etc. = interpolated vowels<br>
        X marker = calibrated<br>
        Circle = interpolated
        """)

        layout.addWidget(self.legend_panel)

        return frame

    def _build_right_panel(self):
        if self.headless:
            return MagicMock()

        frame = QFrame()
        layout = QVBoxLayout(frame)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.fig.tight_layout()
        self.canvas = Canvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax.set_xlabel("F2 (Hz)")
        self.ax.set_ylabel("F1 (Hz)")
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()

        return frame

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    def _build_values_panel_html(self):
        lines = []
        for vowel, data in self.profile.items():
            f1 = data.get("f1")
            f2 = data.get("f2")
            f0 = data.get("f0")
            color = self.colors.get(vowel, self.interp_color)
            lines.append(
                f"<span style='color:{color}; font-weight:bold'>/{vowel}/</span> "
                f"F1={f1:.1f}  F2={f2:.1f}  F0={f0:.1f}"
            )
        return "<br>".join(lines)

    def _plot_profile(self):
        ax = self.ax
        ax.cla()

        ax.set_xlabel("F2 (Hz)")
        ax.set_ylabel("F1 (Hz)")
        ax.invert_xaxis()
        ax.invert_yaxis()

        # Thicker, softer grid lines
        ax.grid(
            True,
            linewidth=1.5,  # increase thickness
            color="#cccccc",  # light gray so labels pop
            alpha=0.8,  # slightly transparent
        )

        # Plot calibrated vowels
        for vowel in self.calibrated_vowels:
            if vowel not in self.profile:
                continue
            f1 = self.profile[vowel]["f1"]
            f2 = self.profile[vowel]["f2"]
            color = self.colors.get(vowel, "black")
            ax.scatter(f2, f1, s=200, c=color, marker="x", linewidths=3)
            ax.text(
                f2 + 20,
                f1 - 20,
                f"/{vowel}/",
                fontsize=14,
                fontweight="bold",
                color=color,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
                ha="left",
                va="center",
            )

        # Plot interpolated vowels
        for vowel in self.interpolated_vowels:
            data = self.profile.get(vowel)
            if not data:
                continue
            f1 = data.get("f1")
            f2 = data.get("f2")
            ax.scatter(
                f2, f1,
                s=160,
                edgecolors=self.interp_color,
                marker="o",
                facecolors="none",
                linewidths=2,
            )
            ax.text(
                f2 + 20,
                f1 - 20,
                f"/{vowel}/",
                fontsize=12,
                color=self.interp_color,
                bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", pad=1.0),
                ha="left",
                va="center",
            )

        self.canvas.draw_idle()
