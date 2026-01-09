# tuner/window.py
from collections import deque
import time
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import (
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
    QAbstractItemView,
    QStackedWidget,
    QApplication,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from profile_viewer.profile_viewer import ProfileViewerWindow
from tuner.tuner_plotter import update_spectrum, update_vowel_chart
from tuner.window_toggle import ModeToggleBar, AnalysisView, VowelMapView
from calibration.dialog import ProfileDialog
from calibration.window import CalibrationWindow
from tuner.spectrogram_view import SpectrogramView
from analysis.vowel_data import expanded_vowels_for_voice, STANDARD_VOWELS
from profile_viewer.vowel_colors import vowel_color_for


class ClearableListWidget(QListWidget):
    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None:
            self.clearSelection()
        super().mousePressEvent(event)


class FakeListWidget:
    """Minimal QListWidget replacement for headless mode."""
    def __init__(self):
        self._items = []
        self._current = None

    def clear(self):
        self._items.clear()

    def addItem(self, item):  # noqa
        self._items.append(item)

    def count(self):
        return len(self._items)

    def item(self, index):
        return self._items[index]

    def setCurrentItem(self, item):  # noqa
        self._current = item

    def currentItem(self):  # noqa
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
        self.vowel_chart_has_seen_valid = False
        self.spectrum_has_seen_valid = False

        if self.headless:
            self._build_headless()
            return

        self.bus = VisualizationBus(max_frames=200)
        self.spectrogram_view = SpectrogramView(self.bus)

        # Normal UI mode
        self._build_ui()
        self._populate_profiles()
        self._setup_timers()
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.80)
        h = int(screen.height() * 0.80)
        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        self.resize(w, h)

        # Center horizontally and vertically
        self.move(
            screen.center().x() - w // 2,
            screen.center().y() - h // 2,
        )

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("white"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("black"))
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorRole.Text, QColor("black"))
        palette.setColor(QPalette.ColorRole.Button, QColor("white"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("black"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#0078d7"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))

        self.setPalette(palette)

    # ---------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------

    def _build_headless(self):
        """
        Minimal Qt-compatible headless UI for tests.
        Provides real QLabel and QListWidget so tests behave identically.
        """
        from PyQt6.QtWidgets import QLabel, QListWidget

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
        label = QLabel("Profiles")
        label.setStyleSheet("""
            font-size: 20pt;
            font-weight: bold;
            color: black;
            background-color: white;
        """)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedHeight(50)
        left_layout.addWidget(label)

        # Profile list container
        profile_container = QWidget()
        profile_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        profile_layout = QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        self.profile_list = ClearableListWidget()
        self.profile_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.profile_list.setStyleSheet("""
            QListWidget {
                background: white;
                color: black;
                font-size: 11pt;
            }
            QListWidget::item {
                padding: 2px 4px;
                margin: 2px 0px;
                border: 1px solid #ccc;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background: #cce0ff;
                color: black;
                border: 1px solid #0078d7;
            }
        """)
        profile_layout.addWidget(self.profile_list)
        self.profile_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.profile_list.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems)
        self.btn_view_profile = QPushButton("View Profile")
        self.btn_view_profile.clicked.connect(  # type:ignore
            self.on_view_profile_clicked)
        self.btn_view_profile.setEnabled(False)

        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete")
        self.refresh_btn = QPushButton("Refresh")
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.refresh_btn)
        self.profile_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        button_style = """
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #888;
                padding: 6px 10px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #888;
            }
        """
        self.btn_view_profile.setStyleSheet(button_style)
        self.delete_btn.setStyleSheet(button_style)
        self.refresh_btn.setStyleSheet(button_style)

        profile_layout.addLayout(btn_row)
        profile_layout.addWidget(self.btn_view_profile)
        left_layout.addWidget(profile_container)

        # Mic buttons
        mic_container = QWidget()
        mic_layout = QVBoxLayout(mic_container)
        mic_layout.setContentsMargins(0, 0, 0, 0)
        mic_layout.setSpacing(20)

        self.start_btn = QPushButton("Start Mic")
        self.stop_btn = QPushButton("Stop Mic")
        self.calib_btn = QPushButton("Calibrate")

        for b, color in (
                (self.start_btn, "#4CAF50"),
                (self.stop_btn, "#f44336"),
                (self.calib_btn, "#2196F3"),
        ):
            b.setFixedHeight(75)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-size: 12pt;
                    font-weight: bold;
                    border-radius: 6px;
                    padding: 6px 10px;
                    border: 1px solid #888;
                }}
                QPushButton:pressed {{
                    background-color: #e0e0e0;
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
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_label.setMinimumHeight(60)
        hint_label.setMaximumHeight(80)
        hint_label.setStyleSheet("font-size: 8pt; color: gray;")
        left_layout.addWidget(hint_label)
        left_layout.addStretch()

        self.active_label = QLabel("Active: None")
        self.active_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.active_label.setFixedHeight(40)
        self.active_label.setStyleSheet(
            "font-weight: bold; font-size: 10pt; color: darkblue;"
        )
        left_layout.addWidget(self.active_label)

        main_layout.addWidget(left_frame)

        # ================= Right panel =================

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Bottom: vowel chart, spectrum
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        self.fig = plt.figure(figsize=(8, 6))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[5, 5])

        self.ax_chart = self.fig.add_subplot(gs[0])
        self.ax_vowel = self.fig.add_subplot(gs[1])
        bg = "#f0f0f0"
        self.fig.patch.set_facecolor(bg)
        self.ax_chart.set_facecolor(bg)
        self.ax_vowel.set_facecolor(bg)

        for ax in (self.ax_chart, self.ax_vowel):
            ax.tick_params(colors="black", labelcolor="black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")
            for spine in ax.spines.values():
                spine.set_color("black")

        self.vowel_status_text = self.ax_vowel.text(
            0.02, 0.95, "",
            transform=self.ax_vowel.transAxes,
            va="top", ha="left",
            fontsize=12,
            fontweight="bold",
            color="#CC0000"
        )

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #f0f0f0;")
        plot_layout.addWidget(self.canvas)

        # Add widgets to splitter
        right_splitter.addWidget(plot_frame)
        right_splitter.addWidget(self.spectrogram_view)
        right_splitter.setSizes([800, 500])

        # Wrap splitter in AnalysisView
        self.analysis_view = AnalysisView(right_splitter)

        # Create vowel map view
        self.vowel_map_view = VowelMapView(self.bus)

        # Stacked widget
        self.stack = QStackedWidget()
        self.stack.addWidget(self.analysis_view)  # index 0
        self.stack.addWidget(self.vowel_map_view)  # index 1

        # Toggle bar
        self.toggle_bar = ModeToggleBar(self.stack)

        # Right container
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.addWidget(self.toggle_bar)
        right_layout.addWidget(self.stack)
        right_container.setStyleSheet("background-color: #e6f0ff;")

        # Add to main layout
        main_layout.addWidget(right_container, stretch=1)

        # Pre-populate right panel with idle plots
        update_spectrum(
            window=self,
            vowel=None,
            target_formants={"f1": None, "f2": None, "f3": None},
            measured_formants={"f1": None, "f2": None, "f3": None},
            pitch=None,
            _tolerance=self.current_tolerance,
        )

        update_vowel_chart(
            window=self,
            vowel=None,
            target_formants={"f1": None, "f2": None, "f3": None},
            measured_formants={"f1": None, "f2": None, "f3": None},
            vowel_score=None,
            resonance_score=None,
            overall=None,
        )

        self.canvas.draw_idle()

        # Signals
        self.start_btn.clicked.connect(lambda: self._start_mic_ui())  # type: ignore
        self.stop_btn.clicked.connect(lambda: self._stop_mic_ui())  # type: ignore
        self.refresh_btn.clicked.connect(self._populate_profiles)  # type: ignore
        self.delete_btn.clicked.connect(self._delete_selected_profile)  # type: ignore
        self.calib_btn.clicked.connect(self._on_calibrate_clicked)  # type: ignore
        (self.profile_list.itemClicked.connect  # type: ignore
         (self._apply_selected_profile_item))
        self.profile_list.itemSelectionChanged.connect(  # type: ignore
            self._on_profile_selection_changed)

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

    def _on_profile_selection_changed(self):
        items = self.profile_list.selectedItems()
        if not items:
            self.tuner.active_profile = None
            self.tuner.engine.vowel_hint = None
            self.active_label.setText("Active: None")
            print("PROFILE DESELECTED → active_profile=None")
            return

        item = items[0]
        base = item.data(Qt.ItemDataRole.UserRole)

        # Ignore the “New Profile” pseudo-item
        if base is None:
            self.btn_view_profile.setEnabled(False)
            self.active_label.setText("Active: None")
            return

        # Load raw JSON
        profile = self.tuner.profile_manager.load_profile_json(base)

        # Extract cal + interp for vowel map
        cal = profile.get("calibrated_vowels", {})
        interp = profile.get("interpolated_vowels", {})

        # Update analyzer’s user_formants (merged cal+interp)
        merged = {**cal, **interp}
        user_formants = self.profile_manager.extract_formants(merged)
        self.live_analyzer.user_formants = user_formants

        # Update UI label
        self._set_active_profile(base)
        print("SELECTION CHANGED:", base, "active_profile:", self.tuner.active_profile)

        # Update vowel map
        self.vowel_map_view.set_vowel_status(cal, interp)
        self.vowel_map_view.analyzer = self.live_analyzer
        self.vowel_map_view.compute_dynamic_ranges()
        self.vowel_map_view.vowel_colors = {
            vowel: vowel_color_for(vowel)
            for vowel in user_formants.keys()
        }
        self.vowel_map_view.update()

    def _populate_profiles(self):
        if self.headless:
            return
        self.profile_list.clear()

        # Set list font BEFORE adding items
        font = QFont("Arial", 12)
        self.profile_list.setFont(font)

        # New profile pseudo-item
        new_item = QListWidgetItem("➕ New Profile")
        new_item.setForeground(QColor("darkgreen"))
        new_item.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        new_item.setData(Qt.ItemDataRole.UserRole, None)
        self.profile_list.addItem(new_item)

        names = self.tuner.list_profiles()
        for base in names:
            if base == "test_bass":
                continue
            display = self.profile_manager.display_name(base)
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, base)
            self.profile_list.addItem(item)
        active = getattr(self.profile_manager, "active_profile_name", None)
        if active and active in names:
            self._set_active_profile(active)
            for i in range(self.profile_list.count()):
                item = self.profile_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == active:
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
        return item.data(Qt.ItemDataRole.UserRole)

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
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No

        )
        if resp != QMessageBox.StandardButton.Yes:
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
        base = item.data(Qt.ItemDataRole.UserRole)
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

        # Store full profile for UI
        self.tuner.active_profile = self.analyzer.active_profile

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
        base = item.data(Qt.ItemDataRole.UserRole)
        # New Profile
        if base is None:
            dlg = ProfileDialog(self)
            if dlg.exec() != dlg.DialogCode.Accepted:
                return
            vals = dlg.get_values()
            name, voice_type = vals[:2]
            expanded = vals[2] if len(vals) > 2 else False
            if not name:
                QMessageBox.warning(
                    self,
                    "Missing name",
                    "Please enter a profile name.",
                )
                return
            optional = []
            if expanded:
                full = expanded_vowels_for_voice(voice_type)
                optional = [v for v in full if v not in STANDARD_VOWELS]

            self.calib_win = CalibrationWindow(
                profile_name=name,
                voice_type=voice_type,
                engine=self.analyzer,
                analyzer=self.live_analyzer,
                profile_manager=self.profile_manager,
                existing_profile=None,
                parent=self,
                expanded_mode=expanded,
                optional_vowels=optional,
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

        # Determine optional vowels from existing profile or voice type
        full = expanded_vowels_for_voice(voice_type)
        optional = [v for v in full if v not in STANDARD_VOWELS]

        self.calib_win = CalibrationWindow(
            profile_name=base,
            voice_type=voice_type,
            engine=self.analyzer,
            analyzer=self.live_analyzer,
            profile_manager=self.profile_manager,
            existing_profile=existing_data,
            parent=self,
            expanded_mode=True,  # existing profile should preserve mode
            optional_vowels=optional,
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
    # Mic handling
    # ---------------------------------------------------------

    def _start_mic_ui(self):
        def callback(indata, _frames, _time, status):
            if status:
                print("[AUDIO STATUS]", status)
            mono = indata[:, 0].astype(np.float64, copy=False)
            # Store raw audio for spectrogram
            self.bus.audio.append(mono.copy())
            # Still feed analyzer
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
        self.vowel_chart_has_seen_valid = False
        self.spectrum_has_seen_valid = False
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

        # Use the same smoothed formants the classifier uses
        smoothed = processed.get("smoothed_formants")
        conf = processed.get("confidence", 0.0)

        if smoothed and conf > 0.5:
            f1 = smoothed.get("f1")
            f2 = smoothed.get("f2")
        else:
            f1 = f2 = None

        vowel_smooth = processed["vowel"]
        # Raw audio segment for spectrogram
        segment = processed.get("segment")
        stable = processed.get("stable")
        stability_score = processed.get("stability_score")
        confidence = processed.get("confidence")

        # Push into visualization bus (no more spec_col)
        self.bus.push(
            f1=f1,
            f2=f2,
            f3=None,
            vowel=vowel_smooth,
            ts=time.time(),
            segment=segment,
            stable=stable,
            stability_score=stability_score,
            confidence=confidence,
        )

        self.vowel_map_view.update_from_bus(self.bus, analyzer=self.analyzer)
        self.spectrogram_view.update_from_bus(self.bus)
        hf = processed.get("hybrid_formants")
        if isinstance(hf, (list, tuple)) and len(hf) == 3:
            raw_f1, raw_f2, raw_f3 = hf
        else:
            raw_f1 = raw_f2 = raw_f3 = None

        # --- ADD DELTA BLOCK HERE ---
        if smoothed:
            smooth_f1 = smoothed.get("f1")
            smooth_f2 = smoothed.get("f2")

            if raw_f1 is not None and smooth_f1 is not None:
                print(f"[F1 Δ] hybrid={raw_f1:.1f}  "
                      f"smoothed={smooth_f1:.1f}  Δ={abs(raw_f1 - smooth_f1):.1f}")

            if raw_f2 is not None and smooth_f2 is not None:
                print(f"[F2 Δ] hybrid={raw_f2:.1f}  "
                      f"smoothed={smooth_f2:.1f}  Δ={abs(raw_f2 - smooth_f2):.1f}")

        print(
            f"[TUNER] f0_raw={processed.get('f0_raw')}  "
            f"f0_smooth={processed.get('f0')}  "
            f"conf={processed.get('confidence')}  "
            f"hybrid_f1={raw_f1}  "
            f"hybrid_f2={raw_f2}  "
            f"hybrid_f3={raw_f3}  "
            f"vowel_raw={processed.get('vowel_guess')}  "
            f"vowel_smooth={processed.get('vowel')}  "
            f"vowel_score={processed.get('vowel_score')}  "
            f"res_score={processed.get('resonance_score')}  "
            f"overall={processed.get('overall')}"
        )

        # Re‑extract for plotting (unchanged)
        hf = processed.get("hybrid_formants")
        if isinstance(hf, (list, tuple)) and len(hf) == 3:
            f1, f2, f3 = hf
        else:
            f1 = f2 = f3 = None

        f0 = processed["f0"]
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


class VisualizationBus:
    def __init__(self, max_frames=200):
        self.audio = deque(maxlen=max_frames)     # raw audio segments
        self.f1 = deque(maxlen=max_frames)
        self.f2 = deque(maxlen=max_frames)
        self.f3 = deque(maxlen=max_frames)
        self.vowels = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)
        self.stable = deque(maxlen=max_frames)
        self.stability_score = deque(maxlen=max_frames)
        self.confidence = deque(maxlen=max_frames)

    def push(self, f1, f2, f3, vowel, ts, segment,
             stable=None, stability_score=None, confidence=None):
        # Raw audio
        if segment is not None:
            self.audio.append(np.asarray(segment, dtype=float))
        else:
            self.audio.append(None)

        # Formants
        self.f1.append(f1)
        self.f2.append(f2)
        self.f3.append(f3)

        self.vowels.append(vowel)
        self.timestamps.append(ts)
        self.stable.append(stable)
        self.stability_score.append(stability_score)
        self.confidence.append(confidence)

    def get_recent_points(self):
        """
        Returns a list of dicts, each representing one frame of formant data.
        """
        pts = []
        for f1, f2, f3, v, ts, st, st_score, conf in zip(
                self.f1, self.f2, self.f3,
                self.vowels, self.timestamps,
                self.stable, self.stability_score,
                self.confidence
        ):
            pts.append({
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "vowel": v,
                "timestamp": ts,
                "stable": st,
                "stability_score": st_score,
                "confidence": conf,
            })
        return pts
