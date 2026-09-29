"""Shared input widgets for the desktop workspace."""
from PySide6.QtWidgets import QComboBox


class WheelSafeComboBox(QComboBox):
    """Keep wheel scrolling from accidentally changing a selected parameter."""

    def wheelEvent(self, event):
        # Ignoring lets Qt route the wheel to the surrounding scroll area.
        # Mouse clicks, popup navigation and keyboard selection remain native.
        event.ignore()
