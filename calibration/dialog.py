# calibration/dialog.py

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox
)


class ProfileDialog(QDialog):
    """
    Simple dialog to collect a profile name, voice type,
    and optional expanded calibration mode.
    """

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
        self.expanded_checkbox.setChecked(False)
        layout.addWidget(self.expanded_checkbox)

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
