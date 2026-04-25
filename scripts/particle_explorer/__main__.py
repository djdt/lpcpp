from particle_explorer.gui import ExplorerWindow

from PySide6 import QtWidgets

from pathlib import Path

if __name__ == "__main__":
    app = QtWidgets.QApplication()
    app.setApplicationName("Particle Explorer")

    win = ExplorerWindow(Path("/home/tom/Downloads/particles.csv"))

    win.show()
    app.exec()
