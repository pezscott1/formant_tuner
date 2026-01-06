# calibration/dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QGroupBox
)
from analysis.vowel_data import expanded_vowels_for_voice

MANDATORY = {"i", "ɛ", "ɑ", "ɔ", "u"}


class ProfileDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("New Profile")
        layout = QVBoxLayout(self)
        # Name field
        layout.addWidget(QLabel("Profile name:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)
        # Voice type dropdown
        layout.addWidget(QLabel("Voice type:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["bass", "baritone", "tenor", "alto", "soprano"])
        layout.addWidget(self.voice_combo)
        # Expanded mode checkbox
        self.expanded_checkbox = QCheckBox("Enable expanded calibration mode")
        layout.addWidget(self.expanded_checkbox)
        # Optional vowel group
        self.optional_group = QGroupBox("Optional vowels to calibrate")
        self.optional_layout = QVBoxLayout()
        self.optional_group.setLayout(self.optional_layout)
        self.optional_group.setVisible(False)
        layout.addWidget(self.optional_group)
        # React to expanded mode toggle
        self.expanded_checkbox.toggled.connect(self.optional_group.setVisible)  # type:ignore
        # React to voice type changes
        self.voice_combo.currentTextChanged.connect(self.populate_optional_vowels)  # type:ignore
        # Initial population
        self.optional_checkboxes = {}
        self.populate_optional_vowels(self.voice_combo.currentText())

        # Buttons
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        ok_btn.clicked.connect(self.accept)   # type:ignore
        cancel_btn.clicked.connect(self.reject)  # type:ignore

    def get_values(self):
        """
        Return (name, voice_type, expanded_mode).
        Existing callers can ignore the third value if they want.
        """
        return (
            self.name_edit.text().strip(),
            self.voice_combo.currentText(),
            self.expanded_checkbox.isChecked(),
        )

    def populate_optional_vowels(self, voice_type):
        # Clear old checkboxes
        while self.optional_layout.count():
            item = self.optional_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Clear dictionary too
        self.optional_checkboxes.clear()

        # Get expanded vowels for this voice
        vowels = expanded_vowels_for_voice(voice_type)
        for v in vowels:
            if v not in MANDATORY:
                cb = QCheckBox(v)
                cb.setChecked(True)
                self.optional_layout.addWidget(cb)
                self.optional_checkboxes[v] = cb

    def selected_optional_vowels(self):
        if not self.expanded_checkbox.isChecked():
            return []
        return [v for v, cb in self.optional_checkboxes.items() if cb.isChecked()]
