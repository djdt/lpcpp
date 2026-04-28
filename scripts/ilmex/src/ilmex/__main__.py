import argparse
from pathlib import Path
from importlib.metadata import version
from PySide6 import QtWidgets

from ilmex.gui import ExplorerWindow


def main():
    parser = argparse.ArgumentParser(
        "ilmex", description="Explorer for inline microscopy data."
    )
    parser.add_argument(
        "path", type=Path, help="path to csv export from lpcpp or l.p.c."
    )

    args = parser.parse_args()

    app = QtWidgets.QApplication()
    app.setApplicationName("ilmex")
    app.setApplicationVersion(version("ilmex"))

    win = ExplorerWindow(args.path)

    win.show()
    app.exec()


if __name__ == "__main__":
    main()
