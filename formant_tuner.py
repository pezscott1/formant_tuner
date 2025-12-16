#!/usr/bin/env python3
import os
import json
import traceback
import numpy as np
import logging
import threading
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QSlider,
    QSplitter,
    QFrame,
    QMessageBox,
    QDialog,
    QSizePolicy,
    QListWidgetItem,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from vowel_data import FORMANTS, NOTE_NAMES
from formant_utils import is_plausible_formants
from mic_analyzer import MicAnalyzer, results_queue
from calibration import CalibrationWindow, ProfileDialog
from voice_analysis import MedianSmoother, PitchSmoother, Analyzer
import sounddevice as sd

matplotlib.use("Qt5Agg")
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)


def profile_display_name(base: str) -> str:
    return base.replace("_", " ")


def profile_files():
    return sorted(
        [
            fn[: -len("_profile.json")]
            for fn in os.listdir(PROFILES_DIR)
            if fn.endswith("_profile.json")
        ]
    )


def profile_base_from_display(display: str):
    if display.startswith("➕"):
        return None
    return display.replace(" ", "_")


def freq_to_note_name(freq: float) -> str:
    if not freq or freq <= 0:
        return "N/A"
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    if midi < 0 or midi >= 128:
        return "N/A"
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


class FormantTunerApp(QMainWindow):
    def __init__(self, analyzer: Analyzer):
        super().__init__()
        self.setWindowTitle("Formant Tuner")
        self.analyzer = analyzer
        self.active_profile = None
        self.calib_win = None
        self.formant_smoother = None
        self.pitch_smoother = None
        # Defaults
        self.voice_type = analyzer.voice_type or "bass"
        self.current_vowel_name = (
            "a"
            if "a" in FORMANTS[self.voice_type]
            else list(FORMANTS[self.voice_type].keys())[0]
        )
        self.current_formants = FORMANTS[self.voice_type][self.current_vowel_name]
        self.last_measured = (np.nan, np.nan, np.nan)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel
        left_frame = QFrame()
        left_frame.setMinimumWidth(250)
        left_frame.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_frame)
        left_frame.setStyleSheet(
            "QFrame { background-color: #f2f2f2; border: 1px solid #ccc; "
            "border-radius: 6px; padding: 8px; }"
        )

        label = QLabel("Profiles")
        label.setStyleSheet("font-size: 10pt; font-weight: bold; margin-bottom: 4px;")
        label.setFixedHeight(50)
        left_layout.addWidget(label)

        # Profile list
        profile_container = QWidget()
        profile_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        profile_layout = QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(6)

        self.profile_list = QListWidget()
        self.profile_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.profile_list.setStyleSheet(
            "QListWidget { font-size: 11pt; padding: 4px; border: 1px solid #ccc; border-radius: 4px; }"
        )
        profile_layout.addWidget(self.profile_list)

        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete")
        self.refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.refresh_btn)
        profile_layout.addLayout(btn_row)

        left_layout.addWidget(profile_container)

        # Mic buttons
        mic_container = QWidget()
        mic_layout = QVBoxLayout(mic_container)
        mic_layout.setContentsMargins(0, 0, 0, 0)
        mic_layout.setSpacing(20)
        mic_layout.addStretch()

        self.start_btn = QPushButton("Start Mic")
        self.stop_btn = QPushButton("Stop Mic")
        self.calib_btn = QPushButton("Calibrate")

        for b, color in (
            (self.start_btn, "#4CAF50"),
            (self.stop_btn, "#f44336"),
            (self.calib_btn, "#2196F3"),
        ):
            b.setFixedHeight(75)
            b.setStyleSheet(
                f"QPushButton {{ background-color: {color}; color: white; font-size: 12pt; "
                "font-weight: bold; border-radius: 6px; padding: 6px; }}"
            )
            mic_layout.addWidget(b)

        left_layout.addWidget(mic_container)

        hint_label = QLabel(
            "Tip: To create a new profile, click Calibrate with New Profile highlighted.\n"
            "To update an existing profile, highlight it first, then click Calibrate."
        )
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet("font-size: 9pt; color: gray;")
        mic_layout.addWidget(hint_label)

        left_layout.addStretch()

        self.active_label = QLabel("Active: —")
        self.active_label.setAlignment(Qt.AlignCenter)
        self.active_label.setFixedHeight(150)
        self.active_label.setStyleSheet(
            "font-weight: bold; font-size: 11pt; color: darkblue;"
        )
        left_layout.addWidget(self.active_label)

        main_layout.addWidget(left_frame)

        # Right side
        right_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(right_splitter, stretch=1)

        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)

        control_layout.addWidget(QLabel("Pitch (Hz)"))
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setMinimum(100)
        self.pitch_slider.setMaximum(600)
        self.pitch_slider.setValue(261)
        control_layout.addWidget(self.pitch_slider)

        self.play_btn = QPushButton("Play Pitch")
        control_layout.addWidget(self.play_btn)

        control_layout.addWidget(QLabel("Tolerance (Hz)"))
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setMinimum(10)
        self.tol_slider.setMaximum(200)
        self.tol_slider.setValue(50)
        control_layout.addWidget(self.tol_slider)

        self.spec_btn = QPushButton("Spectrogram")
        self.spec_btn.setCheckable(True)
        control_layout.addWidget(self.spec_btn)

        right_splitter.addWidget(control_frame)

        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)

        self.fig, (self.ax_chart, self.ax_spec) = plt.subplots(
            1, 2, figsize=(8, 4), constrained_layout=True
        )
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        right_splitter.addWidget(plot_frame)
        right_splitter.setSizes([120, 800])

        self.mic = MicAnalyzer(
            vowel_provider=lambda: self.current_vowel_name,
            tol_provider=lambda: self.tol_slider.value(),
            pitch_provider=lambda: self.pitch_slider.value(),
            sample_rate=44100,
            frame_ms=40,
            analyzer=self.analyzer,
        )

        if hasattr(self.mic, "sample_rate") and self.mic.sample_rate != 44100:
            logger.warning(
                "MicAnalyzer sample_rate differs from UI default: %s",
                self.mic.sample_rate,
            )

        # Signals
        self.pitch_slider.valueChanged.connect(self.on_pitch_change)  # type:ignore
        self.tol_slider.valueChanged.connect(self.on_tol_change)  # type:ignore
        self.play_btn.clicked.connect(
            partial(self.play_pitch, self.pitch_slider.value())
        )  # type:ignore
        self.spec_btn.toggled.connect(self.toggle_spectrogram)  # type:ignore
        self.start_btn.clicked.connect(self.mic.start)  # type:ignore
        self.stop_btn.clicked.connect(self.mic.stop)  # type:ignore
        self.refresh_btn.clicked.connect(self.refresh_profiles)  # type:ignore
        self.delete_btn.clicked.connect(self.delete_profile)  # type:ignore
        self.calib_btn.clicked.connect(self.launch_calibration)  # type:ignore
        self.profile_list.itemDoubleClicked.connect(  # type:ignore
            partial(self.apply_selected_profile)
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_queue)  # type:ignore
        self.timer.start(100)

        # Initial chart build
        self.build_vowel_chart()
        self.update_spectrum(
            self.current_vowel_name,
            self.current_formants,
            (np.nan, np.nan, np.nan),
            pitch=float(self.pitch_slider.value()),
            tolerance=int(self.tol_slider.value()),
        )
        self.canvas.draw()

        # Initial window size ~75% of screen and centered
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.75)
        h = int(screen.height() * 0.75)
        self.resize(w, h)
        self.move(screen.center().x() - w // 2, screen.center().y() - h // 2)

        # Populate profiles initially
        self.refresh_profiles()
        self.show()

    # ----------------- Helper -----------------
    def set_active_profile(self, profile_name: str):
        """Mark a profile as active and update the UI label."""
        self.active_profile = profile_name
        self.active_label.setText(f"Active: {profile_name}")

    # ----------------- UI Actions -----------------
    def on_pitch_change(self, value: int):
        self.update_spectrum(
            self.current_vowel_name,
            self.current_formants,
            self.last_measured,
            float(value),
            int(self.tol_slider.value()),
        )
        self.canvas.draw()

    def on_tol_change(self, value: int):
        self.update_spectrum(
            self.current_vowel_name,
            self.current_formants,
            self.last_measured,
            float(self.pitch_slider.value()),
            int(value),
        )
        self.canvas.draw()

    def toggle_spectrogram(self, checked: bool):
        if checked:
            self.ax_spec.clear()
            self.ax_spec.set_title("Spectrogram (toggle on)")
            self.canvas.draw()
        else:
            self.update_spectrum(
                self.current_vowel_name,
                self.current_formants,
                self.last_measured,
                float(self.pitch_slider.value()),
                int(self.tol_slider.value()),
            )
            self.canvas.draw()

    @staticmethod
    def play_pitch(frequency, duration=2.0, sample_rate=44100):
        def _play():
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            waveform = 0.2 * np.sin(2 * np.pi * frequency * t)
            sd.play(waveform, sample_rate)
            sd.wait()

        threading.Thread(target=_play, daemon=True).start()

    # ----------------- Profiles -----------------
    def refresh_profiles(self):
        self.profile_list.clear()
        new_item = QListWidgetItem("➕ New Profile")
        new_item.setForeground(Qt.darkGreen)
        new_item.setFont(QFont("Consolas", 11, QFont.Bold))
        self.profile_list.addItem(new_item)

        for base in profile_files():
            self.profile_list.addItem(profile_display_name(base))

    def get_selected_profile_base(self):
        item = self.profile_list.currentItem()
        if not item:
            return None
        return profile_base_from_display(item.text())

    def delete_profile(self):
        base = self.get_selected_profile_base()
        if not base:
            QMessageBox.information(self, "Delete", "Select a profile to delete.")
            return
        display = profile_display_name(base)
        path = os.path.join(PROFILES_DIR, f"{base}_profile.json")
        ok = QMessageBox.question(
            self,
            "Delete",
            f"Delete profile {display}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ok == QMessageBox.Yes:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:  # noqa: E722
                traceback.print_exc()
                QMessageBox.critical(self, "Error", "Failed to delete profile.")
        self.refresh_profiles()

    def apply_selected_profile(self):
        item = self.profile_list.currentItem()
        if not item:
            QMessageBox.information(self, "Apply", "Please select a profile to apply.")
            return
        if item.text().startswith("➕"):
            QMessageBox.information(
                self, "New Profile", "Click Calibrate to create a new profile."
            )
            return
        base = profile_base_from_display(item.text())

        with open(
            os.path.join(PROFILES_DIR, "active_profile.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump({"active": base}, fh)

        profile_path = os.path.join(PROFILES_DIR, f"{base}_profile.json")
        model_path = profile_path.replace("_profile.json", "_model.pkl")
        try:
            self.analyzer.load_profile(profile_path, model_path=model_path)
        except Exception:  # noqa: E722
            traceback.print_exc()

        self.voice_type = self.analyzer.voice_type or self.voice_type
        vowels_map = FORMANTS.get(self.voice_type, FORMANTS["bass"])
        self.current_vowel_name = "a" if "a" in vowels_map else next(iter(vowels_map))
        self.current_formants = vowels_map[self.current_vowel_name]
        self.active_label.setText(f"Active: {profile_display_name(base)}")

        self.build_vowel_chart()
        self.update_spectrum(
            self.current_vowel_name,
            self.current_formants,
            self.last_measured,
            float(self.pitch_slider.value()),
            int(self.tol_slider.value()),
        )
        self.canvas.draw()

    def launch_calibration(self):
        item = self.profile_list.currentItem()
        if item and item.text().startswith("➕"):
            dlg = ProfileDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                name, voice_type = dlg.get_values()
                if not name:
                    name = "user1"
                self.calib_win = CalibrationWindow(self.analyzer, name, voice_type)
                self.calib_win.show()
                self.calib_win.destroyed.connect(self.refresh_profiles)
        else:
            base = self.get_selected_profile_base()
            if base:
                voice_type = self.analyzer.voice_type or "bass"
                self.calib_win = CalibrationWindow(self.analyzer, base, voice_type)
                self.calib_win.show()
                self.calib_win.destroyed.connect(self.refresh_profiles)

    # ----------------- Vowel Chart -----------------
    def build_vowel_chart(self):
        self.ax_chart.clear()
        vowels = FORMANTS.get(self.voice_type, FORMANTS["bass"])
        for v, (F1, F2, F3, FS) in vowels.items():
            self.ax_chart.scatter(F2, F1, c="blue", s=70, label=f"/{v}/")
            self.ax_chart.text(F2 + 35, F1 + 35, f"/{v}/", fontsize=9, color="blue")

        self.ax_chart.set_title(f"Vowel Chart ({self.voice_type})")
        self.ax_chart.set_xlabel("F2 (Hz)")
        self.ax_chart.set_ylabel("F1 (Hz)")
        self.ax_chart.invert_xaxis()
        self.ax_chart.invert_yaxis()

    # ----------------- Spectrum -----------------
    def update_spectrum(
        self, vowel, target_formants, measured_formants, pitch, tolerance
    ):
        self.ax_spec.clear()

        # Harmonic series up to 4000 Hz
        hs = [h for h in np.arange(1, 13) * pitch if h <= 4000]
        if not hs:
            self.ax_spec.set_xlim(0, 4000)
            self.ax_spec.set_ylim(0, 1)
            self.ax_spec.set_title("No harmonics in range")
            return

        amps = []
        for h in hs:
            boost = 1.0
            for f in target_formants[:3]:
                if f and not np.isnan(f) and abs(h - f) <= tolerance:
                    boost += 2.0
            amps.append(boost)
        amps = np.array(amps)

        self.ax_spec.stem(hs, amps, linefmt="gray", markerfmt="o", basefmt=" ")

        freq_axis = np.linspace(0, 4000, 1000)
        env = np.zeros_like(freq_axis)
        for f in target_formants[:3]:
            if f and not np.isnan(f):
                env += np.exp(-0.5 * ((freq_axis - f) / 100.0) ** 2)
        self.ax_spec.plot(freq_axis, env, "r-", linewidth=2, label="Filter Envelope")

        for f in target_formants[:3]:
            if f and not np.isnan(f):
                self.ax_spec.axvline(f, color="blue", linestyle="--", alpha=0.5)

        for f in measured_formants[:3]:
            if f and not np.isnan(f):
                self.ax_spec.axvline(f, color="red", linestyle=":", alpha=0.7)

        note = freq_to_note_name(pitch)
        self.ax_spec.set_xlim(0, 4000)
        self.ax_spec.set_ylim(0, max(amps) + 1)
        self.ax_spec.set_title(
            f"Spectrum /{vowel}/ ({self.voice_type}, {note} {pitch:.2f} Hz)"
        )
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Amplitude (a.u.)")

        handles, labels = self.ax_spec.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        if unique:
            self.ax_spec.legend(unique.values(), unique.keys(), loc="upper right")

        f1, f2 = measured_formants[:2]
        if f1 and f2 and not np.isnan(f1) and not np.isnan(f2):
            for a in getattr(self.ax_chart, "_measured_overlay", []):
                try:
                    a.remove()
                except Exception:  # noqa: E722
                    pass
            self.ax_chart._measured_overlay = []
            p = self.ax_chart.scatter(f2, f1, c="red", s=60, zorder=10)
            self.ax_chart._measured_overlay.append(p)

        self.last_measured = measured_formants

    # ----------------- Queue Polling -----------------
    def poll_queue(self):
        if not hasattr(self, "canvas") or self.canvas is None:
            return

        if not hasattr(self, "formant_smoother"):
            self.formant_smoother = MedianSmoother(size=5)
        if not hasattr(self, "pitch_smoother"):
            self.pitch_smoother = PitchSmoother(size=5)

        updated = False
        while not results_queue.empty():
            raw = results_queue.get_nowait()

            f0 = float(raw.get("f0") or self.pitch_slider.value())
            f0 = self.pitch_smoother.update(f0)

            f1, f2, f3 = raw.get("formants", (np.nan, np.nan, np.nan))
            f1, f2, f3 = self.formant_smoother.update(f1, f2, f3)

            ok, reason = is_plausible_formants(f1, f2, self.voice_type)
            if not ok:
                f1, f2 = np.nan, np.nan

            measured = (
                f1 if f1 is not None else np.nan,
                f2 if f2 is not None else np.nan,
                f3 if f3 is not None else np.nan,
            )

            self.update_spectrum(
                raw.get("vowel_guess") or self.current_vowel_name,
                self.current_formants,
                measured,
                f0,
                int(self.tol_slider.value()),
            )

            fb1, fb2 = raw.get("fb_f1"), raw.get("fb_f2")
            self.ax_chart.set_title(
                f"Guess=/{raw.get('vowel_guess')}/ conf={raw.get('vowel_confidence'):.2f}\n"
                f"Feedback: {fb1 or ''} {fb2 or ''}"
            )

            updated = True

        if updated:
            self.canvas.draw()

    # ----------------- Close Handling -----------------
    def closeEvent(self, event):
        try:
            self.timer.stop()
            if hasattr(self.mic, "stop"):
                self.mic.stop()
        except Exception:  # noqa: E722
            logger.exception("Error during shutdown")
        event.accept()


# ----------------- Bootstrap -----------------
def main():
    app = QApplication(sys.argv)
    analyzer = Analyzer(voice_type="bass", smoothing=True, smooth_size=5)
    _win = FormantTunerApp(analyzer)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
