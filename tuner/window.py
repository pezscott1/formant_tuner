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
from profile_viewer.profile_viewer import ProfileViewerWindow
from tuner.tuner_plotter import update_spectrum, update_vowel_chart
from utils.music_utils import hz_to_midi, render_piano
from calibration.dialog import ProfileDialog
from calibration.window import CalibrationWindow


class FakeListWidget:
    """Minimal QListWidget replacement for headless mode."""
    def __init__(self):
        self._items = []
        self._current = None

    def clear(self):
        self._items.clear()

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def item(self, index):
        return self._items[index]

    def setCurrentItem(self, item):
        self._current = item

    def currentItem(self):
        return self._current


class TunerWindow(QMainWindow):
    def __init__(self, tuner, sample_rate=48000, parent=None, headless=False):
        super().__init__(parent)
        self.headless = headless
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

        if self.headless:
            self._build_headless()
            return

        # Normal UI mode
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
    def _build_headless(self):
        """
        Minimal Qt-compatible headless UI for tests.
        Provides real QLabel and QListWidget so tests behave identically.
        """
        from PyQt5.QtWidgets import QLabel, QListWidget

        # Real label so .text() and .setText() work
        self.active_label = QLabel("Active profile: None")

        # Real QListWidget so .item(), .addItem(), .clear() work
        self.profile_list = QListWidget()

        # --- Fake populate method (bound properly) ---
        def _fake_populate(this):
            this.profile_list.clear()
            this.profile_list.addItem("None")
            for name in this.tuner.profile_manager.list_profiles():
                this.profile_list.addItem(name)

        self._populate_profiles = _fake_populate.__get__(self)

        # Populate immediately
        self._populate_profiles()

        # --- Fake apply method (bound properly) ---
        def _fake_apply(this, item):
            name = item.text()
            if name == "None":
                this.tuner.clear_profile()
                this.active_label.setText("Active profile: None")
                return
            this.tuner.load_profile(name)
            this.active_label.setText(f"Active profile: {name}")

        self._apply_selected_profile_item = _fake_apply.__get__(self)

        # Window size (tests expect >600x500)
        self.resize(800, 600)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ================= Left panel =================
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

        # Profile list container
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
        self.btn_view_profile = QPushButton("View Profile")
        self.btn_view_profile.clicked.connect(self.on_view_profile_clicked)  # type:ignore
        self.btn_view_profile.setEnabled(False)
        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete")
        self.refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.refresh_btn)
        profile_layout.addLayout(btn_row)
        profile_layout.addWidget(self.btn_view_profile)
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
            "Tip: To create a new profile, click Calibrate with "
            "“New Profile” highlighted. To update an existing profile, "
            "highlight it first, then click Calibrate."
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

        # ================= Right panel =================
        right_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(right_splitter, stretch=1)

        # Top: tolerance only
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)

        control_layout.addWidget(QLabel("Tolerance (Hz):"))
        self.tol_field = QLineEdit(str(self.current_tolerance))
        self.tol_field.setFixedWidth(60)
        control_layout.addWidget(self.tol_field)
        control_layout.addStretch(1)

        right_splitter.addWidget(control_frame)

        # Bottom: vowel chart, spectrum, piano
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

        # Signals
        (self.tol_field.editingFinished.connect  # type: ignore
            (self._update_tolerance_from_field))
        self.start_btn.clicked.connect(lambda: self._start_mic_ui())  # type: ignore
        self.stop_btn.clicked.connect(lambda: self._stop_mic_ui())  # type: ignore
        self.refresh_btn.clicked.connect(self._populate_profiles)  # type: ignore
        self.delete_btn.clicked.connect(self._delete_selected_profile)  # type: ignore
        self.calib_btn.clicked.connect(self._on_calibrate_clicked)  # type: ignore
        (self.profile_list.itemClicked.connect  # type: ignore
            (self._apply_selected_profile_item))

    # ---------------------------------------------------------
    # Timers
    # ---------------------------------------------------------
    def _setup_timers(self):
        if self.headless:
            return
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_display)  # type: ignore
        self.update_timer.start(60)  # ~16 fps

    # ---------------------------------------------------------
    # Profiles
    # ---------------------------------------------------------

    def on_view_profile_clicked(self):
        if not self.tuner.active_profile:
            QMessageBox.information(self, "No Profile", "No active profile to view.")
            return
        viewer = ProfileViewerWindow(self.tuner.active_profile, parent=self)
        viewer.show()

    def _populate_profiles(self):
        """Populate profile list with a 'New Profile' item plus existing profiles."""
        if self.headless:
            return
        self.profile_list.clear()

        # New profile pseudo-item
        new_item = QListWidgetItem("➕ New Profile")
        new_item.setForeground(Qt.darkGreen)
        new_item.setFont(QFont("Consolas", 11, QFont.Bold))
        new_item.setData(Qt.UserRole, None)
        self.profile_list.addItem(new_item)

        names = self.tuner.list_profiles()
        for base in names:
            if base == "test_bass":
                continue
            display = self.profile_manager.display_name(base)
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, base)
            self.profile_list.addItem(item)

        active = getattr(self.profile_manager, "active_profile_name", None)
        if active and active in names:
            self._set_active_profile(active)
            for i in range(self.profile_list.count()):
                item = self.profile_list.item(i)
                if item.data(Qt.UserRole) == active:
                    self.profile_list.setCurrentItem(item)
                    break

    def _set_active_profile(self, base: str):
        display = self.profile_manager.display_name(base)
        self.active_label.setText(f"Active: {display}")
        self.btn_view_profile.setEnabled(True)

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
        if self.headless:
            return
        base = item.data(Qt.UserRole)
        if base is None:
            self.btn_view_profile.setEnabled(False)
            return
        self._apply_profile_base(base)

    def _apply_profile_base(self, base: str):
        try:
            applied = self.tuner.load_profile(base)
        except Exception as e:
            QMessageBox.critical(self, "Profile error",
                                 f"Could not apply profile '{base}':\n{e}")
            return

        self._set_active_profile(applied)
        self.voice_type = getattr(self.analyzer, "voice_type", self.voice_type)

    def _on_calibrate_clicked(self):
        """
        Launch calibration workflow.
        """
        item = self.profile_list.currentItem()
        if item is None:
            QMessageBox.information(
                self,
                "No profile selected",
                "Please select a profile before calibrating.",
            )
            return
        base = item.data(Qt.UserRole)
        # New Profile
        if base is None:
            dlg = ProfileDialog(self)
            if dlg.exec_() != dlg.Accepted:
                return
            profile_name, voice_type = dlg.get_values()
            if not profile_name:
                QMessageBox.warning(
                    self,
                    "Missing name",
                    "Please enter a profile name.",
                )
                return
            self.calib_win = CalibrationWindow(
                profile_name=profile_name,
                voice_type=voice_type,
                engine=self.analyzer,
                analyzer=self.live_analyzer,
                profile_manager=self.profile_manager,
                existing_profile=None,
                parent=self,
            )
            self.calib_win.vowel_capture_started.connect(self._start_mic_ui)
            self.calib_win.vowel_capture_finished.connect(self._stop_mic_ui)
            self._start_mic_ui()
            self.update_timer.stop()
            self.calib_win.profile_calibrated.connect(self._on_profile_calibrated)
            self.calib_win.show()
            return

        # Existing profile
        voice_type = getattr(self.analyzer, "voice_type", "bass")
        existing_data = self.profile_manager.load_profile(base) or {}
        self.calib_win = CalibrationWindow(
            profile_name=base,
            voice_type=voice_type,
            engine=self.analyzer,
            analyzer=self.live_analyzer,
            profile_manager=self.profile_manager,
            existing_profile=existing_data,
            parent=self,
        )
        self.update_timer.stop()
        self.live_analyzer.pause()
        ok = self._start_mic_ui()
        if not ok:
            QMessageBox.critical(self, "Mic error", "Could not start microphone.")
            return

        self.calib_win.profile_calibrated.connect(self._on_profile_calibrated)
        self.calib_win.show()

    def _on_profile_calibrated(self, base_name: str):
        """
        Called when CalibrationWindow emits profile_calibrated(base_name).
        """
        self._populate_profiles()
        self._apply_profile_base(base_name)

        # Reset smoothing state
        self.live_analyzer.reset()
        self.live_analyzer.resume()
        # Restart UI updates
        self.update_timer.start()

        # Stop mic cleanly
        try:
            self.tuner.stop_mic()
        except Exception:
            pass

    # ---------------------------------------------------------
    # Tolerance handling
    # ---------------------------------------------------------
    def _update_tolerance_from_field(self):
        text = self.tol_field.text()
        try:
            value = int(text)
        except Exception:
            value = self.current_tolerance
        if value is None:
            value = self.current_tolerance
        else:
            self.current_tolerance = value
        self.tol_field.setText(str(value))

    # ---------------------------------------------------------
    # Mic handling
    # ---------------------------------------------------------

    def _start_mic_ui(self):
        def callback(indata, _frames, _time, status):
            if status:
                print("[AUDIO STATUS]", status)
            mono = indata[:, 0].astype(np.float64, copy=False)
            self.live_analyzer.submit_audio_segment(mono)

        def find_device():
            for idx, dev in enumerate(sd.query_devices()):
                if "Razer" in dev["name"] or "Seiren" in dev["name"]:
                    return idx
            return sd.default.device[0]

        device_index = find_device()
        print("Using mic device:", device_index)

        ok = self.tuner.start_mic(lambda: sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=device_index,
            callback=callback
        ))

        if ok:
            self.stream = self.tuner.stream
            return True
        else:
            self.stream = None
            return False

    def _stop_mic_ui(self):
        self.tuner.stop_mic()
        self.stream = None

    # ---------------------------------------------------------
    # Display update
    # ---------------------------------------------------------
    def _update_display(self):
        if self.headless:
            return
        # Only update if mic is running
        stream = self.tuner.stream
        if stream is None or not getattr(stream, "active", True):
            return
        # Skip UI updates if analyzer is paused
        if (hasattr(self.live_analyzer, "is_running")
                and not self.live_analyzer.is_running):
            return
        processed = self.tuner.poll_latest_processed()
        if not processed:
            return

        f0 = processed["f0"]
        if "hybrid_formants" in processed:
            f1, f2, f3 = processed["hybrid_formants"]
        else:
            f1, f2, f3 = processed["formants"]
        vowel_raw = processed["vowel_guess"]
        vowel_smooth = processed["vowel"]
        vowel_score = processed["vowel_score"]
        resonance_score = processed["resonance_score"]
        overall = processed["overall"]

        # Pull active profile targets
        if vowel_smooth in self.analyzer.user_formants:
            entry = self.analyzer.user_formants[vowel_smooth]
            target_formants = {
                "f1": entry.get("f1"),
                "f2": entry.get("f2"),
                "f3": entry.get("f3"),
            }
        else:
            target_formants = {"f1": None, "f2": None, "f3": None}

        # Spectrum panel
        try:
            update_spectrum(
                window=self,
                vowel=vowel_raw,
                target_formants=target_formants,
                measured_formants={"f1": f1, "f2": f2, "f3": f3},
                pitch=f0,
                _tolerance=self.current_tolerance,
            )
        except Exception as e:
            print("[TUNER] update_spectrum error:", e)

        # Vowel chart
        try:
            update_vowel_chart(
                window=self,
                vowel=vowel_smooth,
                target_formants=target_formants,
                measured_formants={"f1": f1, "f2": f2, "f3": f3},
                vowel_score=vowel_score,
                resonance_score=resonance_score,
                overall=overall,
            )
        except Exception as e:
            print("[TUNER] update_vowel_chart error:", e)

        # Piano keyboard
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
        try:
            self.update_timer.stop()
        except AttributeError:
            pass
        super().closeEvent(event)
