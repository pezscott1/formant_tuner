# tuner/window.py
import threading
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QSplitter,
    QFrame,
    QMessageBox,
    QSizePolicy,
    QListWidgetItem,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from tuner.tuner_plotter import update_spectrum, update_vowel_chart
from utils.music_utils import hz_to_midi, render_piano
from calibration.dialog import ProfileDialog
from calibration.window import CalibrationWindow


class TunerWindow(QMainWindow):
    """
    Restored classic UI + modern right panel.

    Internals:
      - analyzer: FormantAnalysisEngine (tuner.engine)
      - profile_manager: ProfileManager (tuner.profile_manager)
      - live_analyzer: LiveAnalyzer (tuner.live_analyzer)
    """

    def __init__(self, tuner, sample_rate=44100, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Formant Tuner")

        self.tuner = tuner
        self.analyzer = tuner.engine
        self.profile_manager = tuner.profile_manager
        self.live_analyzer = tuner.live_analyzer
        self.sample_rate = sample_rate
        # State
        self.current_tolerance = 50
        self.voice_type = getattr(self.analyzer, "voice_type", "bass")
        self.stream = None
        self.stream_lock = threading.Lock()

        # For tuner_plotter hooks (if needed)
        self.vowel_measured_artist = None
        self.vowel_line_artist = None

        self._build_ui()
        self._populate_profiles()
        self._setup_timers()

        # Initial window size
        screen = self.screen().availableGeometry()
        w = int(screen.width() * 0.75)
        h = int(screen.height() * 0.75)
        self.resize(w, h)
        self.move(screen.center().x() - w // 2, screen.center().y() - h // 2)

    # ---------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ================= Left panel (restored classic look) =================
        left_frame = QFrame()
        left_frame.setMinimumWidth(250)
        left_frame.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_frame)
        left_frame.setStyleSheet(
            "QFrame { background-color: #f2f2f2; border: 1px solid #ccc; "
            "border-radius: 6px; padding: 8px; }"
        )

        label = QLabel("Profiles")
        label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        label.setFixedHeight(50)
        left_layout.addWidget(label)

        # ---- Profile list container ----
        profile_container = QWidget()
        profile_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        profile_layout = QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(6)

        self.profile_list = QListWidget()
        self.profile_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.profile_list.setStyleSheet(
            "QListWidget { font-size: 11pt; padding: 4px; border: 1px solid #ccc; "
            "border-radius: 4px; }"
        )
        profile_layout.addWidget(self.profile_list)

        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete")
        self.refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.refresh_btn)
        profile_layout.addLayout(btn_row)

        left_layout.addWidget(profile_container)

        # ---- Mic buttons ----
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
                f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-size: 12pt;
                    font-weight: bold;
                    border-radius: 6px;
                    padding: 6px;
                }}
                """
            )
            mic_layout.addWidget(b)

        left_layout.addWidget(mic_container)

        hint_label = QLabel(
            "Tip: To create a new profile, click Calibrate with"
            "“New Profile” highlighted. To update an existing"
            "profile, highlight it first, then click Calibrate."
        )
        hint_label.setWordWrap(True)
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet("font-size: 9pt; color: gray;")
        hint_label.setMinimumHeight(150)
        mic_layout.addWidget(hint_label)

        left_layout.addStretch()

        self.active_label = QLabel("Active: None")
        self.active_label.setAlignment(Qt.AlignCenter)
        self.active_label.setFixedHeight(150)
        self.active_label.setStyleSheet(
            "font-weight: bold; font-size: 11pt; color: darkblue;"
        )
        left_layout.addWidget(self.active_label)

        main_layout.addWidget(left_frame)

        # ================= Right panel (modern) =================
        right_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(right_splitter, stretch=1)

        # ---- Top: tolerance only ----
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)

        control_layout.addWidget(QLabel("Tolerance (Hz):"))
        self.tol_field = QLineEdit(str(self.current_tolerance))
        self.tol_field.setFixedWidth(60)
        control_layout.addWidget(self.tol_field)
        control_layout.addStretch(1)

        right_splitter.addWidget(control_frame)

        # ---- Bottom: vowel chart, spectrum, piano ----
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)

        self.fig = plt.figure(figsize=(8, 6))
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.3, bottom=0.12)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 4, 1])

        self.ax_chart = self.fig.add_subplot(gs[0])
        self.ax_vowel = self.fig.add_subplot(gs[1])
        self.ax_piano = self.fig.add_subplot(gs[2])

        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        right_splitter.addWidget(plot_frame)
        right_splitter.setSizes([80, 800])

        # ---- Signals ----
        self.tol_field.editingFinished.connect(  # type:ignore
            self._update_tolerance_from_field)
        self.start_btn.clicked.connect(self.start_mic)  # type:ignore
        self.stop_btn.clicked.connect(self.stop_mic)  # type:ignore
        self.refresh_btn.clicked.connect(self._populate_profiles)  # type:ignore
        self.delete_btn.clicked.connect(self._delete_selected_profile)  # type:ignore
        self.calib_btn.clicked.connect(self._on_calibrate_clicked)  # type:ignore
        self.profile_list.itemClicked.connect(  # type:ignore
            self._apply_selected_profile_item)

    # ---------------------------------------------------------
    # Timers
    # ---------------------------------------------------------
    def _setup_timers(self):
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_display)  # type:ignore
        self.update_timer.start(60)  # ~16 fps

    # ---------------------------------------------------------
    # Profiles
    # ---------------------------------------------------------
    def _populate_profiles(self):
        """Populate profile list with a 'New Profile' item plus existing profiles."""
        self.profile_list.clear()

        # New profile pseudo-item
        new_item = QListWidgetItem("➕ New Profile")
        new_item.setForeground(Qt.darkGreen)
        new_item.setFont(QFont("Consolas", 11, QFont.Bold))
        new_item.setData(Qt.UserRole, None)
        self.profile_list.addItem(new_item)

        names = self.profile_manager.list_profiles()
        for base in names:
            display = self.profile_manager.display_name(base)
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, base)
            self.profile_list.addItem(item)

        active = getattr(self.profile_manager, "active_profile_name", None)
        if active and active in names:
            self._set_active_profile(active)
            # Select it in the list
            for i in range(self.profile_list.count()):
                item = self.profile_list.item(i)
                if item.data(Qt.UserRole) == active:
                    self.profile_list.setCurrentItem(item)
                    break
        elif names:
            # Default to first real profile
            # base = names[0]
            # self._apply_profile_base(base)
            pass

    def _set_active_profile(self, base: str):
        """Update the UI label for the active profile."""
        display = self.profile_manager.display_name(base)
        self.active_label.setText(f"Active: {display}")

    def _get_selected_profile_base(self):
        item = self.profile_list.currentItem()
        if not item:
            return None
        return item.data(Qt.UserRole)

    def _delete_selected_profile(self):
        base = self._get_selected_profile_base()
        if base is None:
            QMessageBox.warning(self, "No Selection",
                                "Please select a profile to delete.")
            return
        display = self.profile_manager.display_name(base)
        resp = QMessageBox.question(
            self,
            "Delete profile",
            f"Delete profile '{display}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return
        try:
            self.profile_manager.delete_profile(base)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to delete profile:\n{e}",
            )
        self._populate_profiles()

    def _apply_selected_profile_item(self, item: QListWidgetItem):
        base = item.data(Qt.UserRole)
        if base is None:
            # "New Profile" double-clicked: leave for calibration workflow
            QMessageBox.information(
                self,
                "New profile",
                "Use Calibrate with “New Profile” selected to create a profile.",
            )
            return
        self._apply_profile_base(base)

    def _apply_profile_base(self, base: str):

        try:
            applied = self.profile_manager.apply_profile(base)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Profile error",
                f"Could not apply profile '{base}':\n{e}",
            )
            return
        # Update UI label
        self._set_active_profile(applied)
        # Load the profile JSON
        try:
            data = self.profile_manager.load_profile_json(applied)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Profile error",
                f"Failed to load profile data for '{applied}':\n{e}",
            )
            return

        # Extract calibrated formants
        formants = self.profile_manager.extract_formants(data)
        # Store in engine so LiveAnalyzer can score vowels
        self.analyzer.calibrated_profile = formants

        # Update engine voice_type if profile changed it
        self.voice_type = getattr(self.analyzer, "voice_type", self.voice_type)

    def _on_calibrate_clicked(self):
        """
        Launch calibration workflow.

        - If “New Profile” is selected → ask for name + voice type
        - If an existing profile is selected → update that profile
        - When calibration finishes → refresh list + apply profile
        """
        item = self.profile_list.currentItem()
        base = item.data(Qt.UserRole) if item else None
        # --- Case 1: New Profile ---
        if base is None:
            dlg = ProfileDialog(self)
            if dlg.exec_() != dlg.Accepted:
                return

            profile_name, voice_type = dlg.get_values()
            if not profile_name:
                QMessageBox.warning(self,
                                    "Missing name", "Please enter a profile name.")
                return

            # Launch calibration window
            self.calib_win = CalibrationWindow(
                profile_name=profile_name,
                voice_type=voice_type,
                analyzer=self.analyzer,
                parent=self,
            )
            self.update_timer.stop()  # pause tuner UI updates
            self.start_mic()
            print("[AUDIO] Microphone stream started")
            self.calib_win.profile_calibrated.connect(self._on_profile_calibrated)
            self.calib_win.show()
            return

        # --- Case 2: Update existing profile ---
        voice_type = getattr(self.analyzer, "voice_type", "bass")

        self.calib_win = CalibrationWindow(
            profile_name=base,
            voice_type=voice_type,
            analyzer=self.analyzer,
            parent=self,
        )
        self.update_timer.stop()
        self.start_mic()
        print("[AUDIO] Microphone stream started")
        self.calib_win.profile_calibrated.connect(self._on_profile_calibrated)
        self.calib_win.show()

    def _on_profile_calibrated(self, base_name: str):
        """
        Called when CalibrationWindow emits profile_calibrated(base_name).
        """
        # Refresh list and select the new/updated profile
        self._populate_profiles()
        self._apply_profile_base(base_name)
        self.live_analyzer.reset()
        # Restart analyzer stream after calibration shut it down
        if hasattr(self.analyzer, "start_stream"):
            try:
                self.analyzer.start_stream()
            except Exception as e:
                print("[AUDIO] Failed to restart analyzer stream:", e)
        self.update_timer.start()  # resume tuner UI updates
        self.stop_mic()

    # ---------------------------------------------------------
    # Tolerance handling
    # ---------------------------------------------------------
    def _update_tolerance_from_field(self):
        text = self.tol_field.text()
        try:
            value = int(text)
            if value <= 0:
                raise ValueError
            self.current_tolerance = value
        except ValueError:
            self.tol_field.setText(str(self.current_tolerance))

    # ---------------------------------------------------------
    # Mic handling
    # ---------------------------------------------------------
    def start_mic(self):
        with self.stream_lock:
            if self.stream is not None:
                return

            def callback(indata, _frames, _time, status):
                if status:
                    print("[AUDIO STATUS]", status)

                mono = indata[:, 0].copy()

                # Feed pitch smoother's audio buffer (via live_analyzer)
                try:
                    self.live_analyzer.pitch_smoother.push_audio(mono)
                except Exception as error:
                    print("[AUDIO BUFFER ERROR]", error)

                # Process frame
                try:
                    self.analyzer.process_frame(mono, self.sample_rate)
                except Exception as error:
                    print("[AUDIO CALLBACK ERROR]", error)

            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=callback,
                )
                self.stream.start()
                print("[AUDIO] Microphone stream started")
            except Exception as e:
                self.stream = None
                QMessageBox.critical(
                    self,
                    "Mic error",
                    f"Could not start microphone:\n{e}",
                )

    def stop_mic(self):
        with self.stream_lock:
            if self.stream is None:
                return
            try:
                self.stream.stop()
                self.stream.close()
                print("[AUDIO] Microphone stream stopped")
            except Exception:
                pass
            finally:
                self.stream = None
                self.live_analyzer.reset()

    # ---------------------------------------------------------
    # Display update
    # ---------------------------------------------------------
    def _update_display(self):  # noqa: C901
        if self.stream is None or not self.stream.active:
            return

        raw = self.analyzer.get_latest_raw()
        if raw is None:
            return

        processed = self.live_analyzer.process_raw(raw)
        if processed is None:
            return
        print(
            "PROCESSED:",
            processed["vowel"],
            processed["vowel_score"],
            processed["resonance_score"],
            processed["overall"],
        )
        # Extract smoothed values
        f0 = processed["f0"]
        f1, f2, f3 = processed["formants"]
        vowel = processed["vowel"]
        vowel_score = processed["vowel_score"]
        resonance_score = processed["resonance_score"]
        overall = processed["overall"]

        # Your plotter expects:
        #   target_formants = (f1_t, f2_t, f3_t)
        #   measured_formants = (f1_m, f2_m, f3_m)
        #
        # For now, we have no target formants in the tuner UI,
        # so we pass (np.nan, np.nan, np.nan)
        target_formants = (np.nan, np.nan, np.nan)
        measured_formants = (f1, f2, f3)
        raw = self.analyzer.get_latest_raw()
        if raw is None:
            return
        # ===== Spectrum panel =====
        try:
            update_spectrum(
                window=self,
                vowel=vowel,
                target_formants=target_formants,
                measured_formants=measured_formants,
                pitch=f0,
                _tol=self.current_tolerance,
            )
        except Exception as e:
            print("[TUNER] update_spectrum error:", e)

        # ===== Vowel chart =====
        try:
            update_vowel_chart(
                window=self,
                vowel=vowel,
                target_formants=target_formants,
                measured_formants=measured_formants,
                vowel_score=vowel_score,
                resonance_score=resonance_score,
                overall=overall,
            )
        except Exception as e:
            print("[TUNER] update_vowel_chart error:", e)

        # ===== Piano keyboard =====
        try:
            self.ax_piano.cla()
            midi_note = hz_to_midi(f0) if (f0 is not None and f0 > 0) else None
            render_piano(self.ax_piano, midi_note)
        except Exception as e:
            print("[TUNER] piano render error:", e)

        self.canvas.draw_idle()

    # ---------------------------------------------------------
    # Close handling
    # ---------------------------------------------------------
    def closeEvent(self, event):
        self.update_timer.stop()
        super().closeEvent(event)
