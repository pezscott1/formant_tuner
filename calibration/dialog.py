# calibration/dialog.py

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton
)


class ProfileDialog(QDialog):
    """
    Simple dialog to collect a profile name and voice type.
    Used when creating a new calibration profile.
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

        # Buttons
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        ok_btn.clicked.connect(self.accept)  # type:ignore
        cancel_btn.clicked.connect(self.reject)  # type:ignore

    def get_values(self):
        """Return (name, voice_type)."""
        return self.name_edit.text().strip(), self.voice_combo.currentText()
