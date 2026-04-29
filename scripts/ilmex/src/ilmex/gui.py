import numpy as np
import numpy.lib.recfunctions as rfn
from importlib.metadata import version

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ilmex.colors import cividis
from ilmex.charts import HistogramChart, ScatterChart
from ilmex.widgets import RangeSlider


def array_to_image(array: np.ndarray) -> QtGui.QImage:
    """Converts a numpy array to a Qt image."""

    if array.dtype in [np.float32, np.float64]:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)

    array = np.ascontiguousarray(array)
    image = QtGui.QImage(
        array.data,
        array.shape[1],
        array.shape[0],
        array.strides[0],
        QtGui.QImage.Format.Format_Indexed8,
    )
    image._array = array  # type: ignore
    return image


class ImageItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(parent)

        self.image = image
        self.rect = rect

    def setImage(self, image: QtGui.QImage):
        self.image = image
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.drawImage(self.rect, self.image)


class ControlSlider(RangeSlider):
    rangeChanged = QtCore.Signal(str, float, float)

    def __init__(
        self,
        name: str,
        scale: float = 1.0,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.name = name
        self.scale = scale

        self.valueChanged.connect(self._rangeChanged)
        self.value2Changed.connect(self._rangeChanged)

        self.action_min = QtGui.QAction("Set Minimum")
        self.action_min.triggered.connect(self.dialogMinimum)
        self.action_max = QtGui.QAction("Set Maximum")
        self.action_max.triggered.connect(self.dialogMaximum)

    def _scale(self, v: float, inverse: bool = False) -> float:
        if inverse:
            v = v / self.scale
        else:
            v = v * self.scale
        if not np.isfinite(v):
            v = 0.0
        return v

    def dialogMinimum(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Set Minimum",
            "Set Minimum",
            self._scale(self.left()),
            self._scale(self.minimum()),
            self._scale(self.maximum()),
        )
        if ok:
            self.setLeft(int(self._scale(val, True)))

    def dialogMaximum(self):
        val, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Set Maximum",
            "Set Maximum",
            self._scale(self.right()),
            self._scale(self.minimum()),
            self._scale(self.maximum()),
        )
        if ok:
            self.setRight(int(self._scale(val, True)))

    def _rangeChanged(self):
        self.rangeChanged.emit(
            self.name, self._scale(self.left()), self._scale(self.right())
        )

    def setRangeScaled(self, min: float, max: float):
        super().setRange(int(self._scale(min, True)), int(self._scale(max, True)))

    def setScaled(self, left: float, right: float):
        super().setValues(int(self._scale(left, True)), int(self._scale(right, True)))

    def scaled(self) -> tuple[float, float]:
        return self._scale(self.left()), self._scale(self.right())

    # def min(self) -> float:
    #     return self.scaled(self.left())
    #
    # def max(self) -> float:
    #     return self.scaled(self.right())

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_min)
        menu.addAction(self.action_max)
        menu.popup(event.globalPos())


class ScatterWidget(QtWidgets.QWidget):
    redrawRequested = QtCore.Signal()

    def __init__(self, data: np.ndarray, parent: QtWidgets.QWidget | None = None):
        assert data.dtype.names is not None
        super().__init__(parent)
        self.chart = ScatterChart()

        self.combo_x = QtWidgets.QComboBox()
        self.combo_x.addItems(list(data.dtype.names))
        self.combo_y = QtWidgets.QComboBox()
        self.combo_y.addItems(list(data.dtype.names))
        self.combo_y.setCurrentIndex(1)

        self.combo_x.currentIndexChanged.connect(self.redrawRequested)
        self.combo_y.currentIndexChanged.connect(self.redrawRequested)

        layout_combo = QtWidgets.QHBoxLayout()
        layout_combo.addWidget(
            QtWidgets.QLabel("x:"), 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        layout_combo.addWidget(self.combo_x, 1)
        layout_combo.addWidget(
            QtWidgets.QLabel("y:"), 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        layout_combo.addWidget(self.combo_y, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chart, 1)
        layout.addLayout(layout_combo, 0)
        self.setLayout(layout)

    def updateScatter(self, data: np.ndarray, reset_roi: bool = True):
        xs = data[self.combo_x.currentText()]
        ys = data[self.combo_y.currentText()]
        self.chart.updateScatter(xs, ys)

    def updateROI(self, data: np.ndarray):
        xs = data[self.combo_x.currentText()]
        ys = data[self.combo_y.currentText()]

        xmin, xmax = np.percentile(xs, [1, 99])
        ymin, ymax = np.percentile(ys, [1, 99])

        self.chart.roi.setPos(xmin, ymin)
        self.chart.roi.setSize(QtCore.QPointF(xmax - xmin, ymax - ymin))

        self.chart.setLimits(xMin=xs.min(), xMax=xs.max(), yMin=ys.min(), yMax=ys.max())


class CapillaryWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.count = QtWidgets.QLabel()

        self.view = QtWidgets.QGraphicsView()
        self.view.setScene(QtWidgets.QGraphicsScene())

        self.image = ImageItem(
            QtGui.QImage(),
            QtCore.QRectF(
                0.0, 0.0, ExplorerWindow.CAMERA_SIZE[0], ExplorerWindow.CAMERA_SIZE[1]
            ),
        )
        self.view.setFixedSize(
            QtCore.QSize(
                ExplorerWindow.CAMERA_SIZE[0] // 25 * 4 + 10,
                ExplorerWindow.CAMERA_SIZE[1] // 25 * 4 + 10,
            )
        )
        self.view.scene().addItem(self.image)
        self.view.setSceneRect(self.image.boundingRect())
        self.view.scale(4.0 / 25.0, 4.0 / 25.0)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view, 1)
        layout.addWidget(self.count, 0)

        self.setLayout(layout)

    def updateImage(self, data: np.ndarray):
        hist, _, _ = np.histogram2d(
            data["y"],
            data["x"],
            bins=(
                np.arange(0, ExplorerWindow.CAMERA_SIZE[1], 25),
                np.arange(0, ExplorerWindow.CAMERA_SIZE[0], 25),
            ),
        )
        vmin, vmax = 0.0, np.percentile(hist, 98)
        if vmax != vmin:
            hist = (np.clip(hist, vmin, vmax) - vmin) / (vmax - vmin)

        array = np.ascontiguousarray(hist * 255, dtype=np.uint8)
        image = QtGui.QImage(
            array.data,
            array.shape[1],
            array.shape[0],
            array.strides[0],
            QtGui.QImage.Format.Format_Indexed8,
        )
        image._array = array  # type: ignore
        image.setColorTable(cividis)
        image.setColorCount(len(cividis))

        self.image.setImage(image)
        self.count.setText(f"Particles: {data.size}")


class ExplorerWindow(QtWidgets.QMainWindow):
    CAMERA_SIZE = 2048, 1536
    VALID_RANGES = {  # name : (min val, max val, scale)
        "frame_count": (1, None, 1),
        "area": (0, None, 1),
        "aspect": (0, 1.0, 1e-2),
        "circularity": (0, 1.0, 1e-2),
        "convexity": (0, 1.0, 1e-2),
        "solidity": (0, 1.0, 1e-2),
        "intensity": (0, None, 1),
        "sharpness": (0, None, 1),
        "x": (None, None, 1),
        "y": (None, None, 1),
    }
    REMAP_NAMES = {
        "circularity_1": "circularity",
        "convexity_1": "covexity",
        "solidity_1": "solidity",
        "particle_diameter_aspect_ratios_1": "aspect",
        "centroid_position_row_pix": "y",
        "centroid_position_column_pix": "x",
        "largest_feret_diameters_µm": "diameter",
    }

    def __init__(
        self,
        path: Path,
        pixel_size: float = 0.46,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.resize(1200, 800)

        self.data, self.data_format = self.importData(path)
        assert self.data.dtype.names is not None

        # fixes for specific inputs
        if self.data_format == "brave":
            self.data = rfn.rename_fields(self.data, ExplorerWindow.REMAP_NAMES)
            self.data["aspect"] = 1.0 / self.data["aspect"]
        elif self.data_format == "lpcpp":  # convert pixels to um
            self.data["area"] *= pixel_size**2
            self.data["circular_equivalent_diameter"] *= pixel_size
            self.data["radius"] *= pixel_size
            self.data = rfn.append_fields(
                self.data, "diameter", self.data["radius"] * 2.0
            )
            print(self.data)
        else:
            raise ValueError("unknown data format:", self.data_format)

        # convert to simple form
        assert self.data.dtype.names is not None

        self.scatter = ScatterWidget(self.data)
        self.scatter.chart.roi.sigRegionChangeFinished.connect(self.redrawCapillary)
        self.scatter.chart.roi.sigRegionChangeFinished.connect(self.redrawHistogram)
        self.scatter.redrawRequested.connect(self.redrawScatter)
        self.scatter.redrawRequested.connect(self.redrawCapillary)

        self.hist = HistogramChart()
        self.hist.setLimits(
            xMin=0.0, xMax=self.data["radius"].max(), yMin=0.0, yMax=100.0
        )
        self.hist.region.setRegion(np.percentile(self.data["radius"], [1, 99]))
        self.hist.region.setBounds((0.0, self.data["radius"].max()))
        self.hist.region.sigRegionChangeFinished.connect(self.redrawCapillary)
        self.hist.region.sigRegionChangeFinished.connect(self.updateScatter)

        self.hist.cursorMoved.connect(self.printCursorPos)

        self.capillary = CapillaryWidget()

        self.sliders = {}
        for name, (vmin, vmax, scale) in ExplorerWindow.VALID_RANGES.items():
            if name not in self.data.dtype.names:
                continue
            if vmin is None:
                vmin = self.data[name].min()
            if vmax is None:
                vmax = self.data[name].max()

            self.sliders[name] = ControlSlider(name, scale=scale)
            self.sliders[name].setRangeScaled(vmin, vmax)
            self.sliders[name].setScaled(vmin, vmax)
            self.sliders[name].rangeChanged.connect(self.redrawAll)
            self.sliders[name].rangeChanged.connect(self.printControl)

        self.status_bar = self.statusBar()

        controls_layout = QtWidgets.QFormLayout()
        for name, slider in self.sliders.items():
            controls_layout.addRow(name.replace("_", " ").title(), slider)

        controls_widget = QtWidgets.QWidget()
        controls_widget.setMinimumWidth(300)
        controls_widget.setLayout(controls_layout)

        capillary_dock = QtWidgets.QDockWidget("Capillary")
        capillary_dock.setWidget(self.capillary)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, capillary_dock)

        controls_dock = QtWidgets.QDockWidget("Controls")
        controls_dock.setWidget(controls_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, controls_dock)

        hist_dock = QtWidgets.QDockWidget("Histogram")
        hist_dock.setWidget(self.hist)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, hist_dock)

        scatter_dock = QtWidgets.QDockWidget("Scatter")
        scatter_dock.setWidget(self.scatter)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, scatter_dock)

        self.tabifyDockWidget(scatter_dock, hist_dock)

        self.resizeDocks(
            [controls_dock, hist_dock], [350, 850], QtCore.Qt.Orientation.Horizontal
        )

        self.createMenuBar()
        self.redrawAll()

    def createMenuBar(self):
        menu_bar = self.menuBar()

        file = menu_bar.addMenu("File")
        action_quit = QtGui.QAction("&Open", parent=self)
        action_quit.triggered.connect(self.dialogImportData)
        action_quit.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        file.addAction(action_quit)

        action_quit = QtGui.QAction("&Save", parent=self)
        action_quit.triggered.connect(self.dialogExportData)
        action_quit.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        file.addAction(action_quit)

        action_quit = QtGui.QAction("&Quit", parent=self)
        action_quit.triggered.connect(self.close)
        action_quit.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        file.addAction(action_quit)

    def dialogImportData(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "CSV Documents (*.csv);;All files (*)"
        )
        if path != "":
            self.importData(path)

    def dialogExportData(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "CSV Documents (*.csv);;All files (*)"
        )
        if path != "":
            if not path.endswith(".csv"):
                path = path + ".csv"
            self.exportFilteredData(path)

    def importData(self, path: str | Path) -> tuple[np.ndarray, str]:
        with open(path) as fp:
            line = fp.readline()
            delimiter = ";" if ";" in line else ","
            format = "brave" if line.startswith("particle id") else "lpcpp"

        return np.genfromtxt(path, names=True, delimiter=delimiter), format

    def exportFilteredData(self, output: str):
        data = self.filteredData(True, True)
        assert data.dtype.names is not None

        with open(output, "w") as fp:
            fp.write(f"# ilmex {version('ilmex')} export\n")
            fp.write(";".join(data.dtype.names) + "\n")
            np.savetxt(fp, data, delimiter=";", fmt="%.12g")

    def printControl(self, name: str, min: float, max: float):
        self.status_bar.showMessage(f"{name}: {min:.12g} - {max:.12g}")

    def printCursorPos(self, p: QtCore.QPointF):
        self.status_bar.showMessage(f"{p.x():.4g}, {p.y():.4g}")

    def filteredData(self, hist: bool = False, scatter: bool = False) -> np.ndarray:
        data = self.data
        for name, slider in self.sliders.items():
            vmin, vmax = slider.scaled()
            data = data[np.logical_and(data[name] >= vmin, data[name] <= vmax)]

        if hist:
            hist_min, hist_max = self.hist.region.getRegion()
            data = data[
                np.logical_and(
                    data["diameter"] >= hist_min, data["diameter"] <= hist_max
                )
            ]
        if scatter:
            xmin, ymin = self.scatter.chart.roi.pos()
            dx, dy = self.scatter.chart.roi.size()
            data = data[
                np.logical_and(
                    data[self.scatter.combo_x.currentText()] >= xmin,
                    data[self.scatter.combo_x.currentText()] <= xmin + dx,
                )
            ]
            data = data[
                np.logical_and(
                    data[self.scatter.combo_y.currentText()] >= ymin,
                    data[self.scatter.combo_y.currentText()] <= ymin + dy,
                )
            ]

        return data

    def redrawAll(self):
        self.redrawHistogram()
        self.redrawCapillary()
        self.redrawScatter()

    def redrawCapillary(self):
        data = self.filteredData(hist=True, scatter=True)
        self.capillary.updateImage(data)

    def redrawHistogram(self):
        data = self.filteredData(scatter=True)
        self.hist.updateHistogram(data["diameter"], bins=100)

    def updateScatter(self):
        data = self.filteredData(hist=True)
        self.scatter.updateScatter(data)

    def redrawScatter(self):
        data = self.filteredData(hist=True)
        self.scatter.updateScatter(data)
        self.scatter.updateROI(data)
