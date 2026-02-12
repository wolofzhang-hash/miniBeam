import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from pathlib import Path
from .ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    icon_path = Path(__file__).resolve().parent / "assets" / "app_icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())
