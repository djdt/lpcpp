import argparse
from pathlib import Path

from PySide6 import QtWidgets


from particle_explorer.gui import ExplorerWindow

if __name__ == "__main__":
    parser = argparse.ArgumentParser("particle_explorer")
    parser.add_argument("path", type=Path, help="path to csv export")

    args = parser.parse_args()

    app = QtWidgets.QApplication()
    app.setApplicationName("Particle Explorer")

    win = ExplorerWindow(args.path)

    win.show()
    app.exec()
