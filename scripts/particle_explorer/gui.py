import numpy as np
import numpy.lib.recfunctions as rfn

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from particle_explorer.colors import cividis
from particle_explorer.charts import HistogramChart, ScatterChart
from particle_explorer.widgets import RangeSlider


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
    updateRequested = QtCore.Signal()

    def __init__(self, data: np.ndarray, parent: QtWidgets.QWidget | None = None):
        assert data.dtype.names is not None
        super().__init__(parent)
        self.chart = ScatterChart()

        self.combo_x = QtWidgets.QComboBox()
        self.combo_x.addItems(list(data.dtype.names))
        self.combo_y = QtWidgets.QComboBox()
        self.combo_y.addItems(list(data.dtype.names))
        self.combo_y.setCurrentIndex(1)

        self.combo_x.currentIndexChanged.connect(self.updateRequested)
        self.combo_y.currentIndexChanged.connect(self.updateRequested)

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

    def updateScatter(self, data: np.ndarray):
        xs = data[self.combo_x.currentText()]
        ys = data[self.combo_y.currentText()]
        self.chart.updateScatter(xs, ys)

    # def filterData(self, data: np.ndarray) -> np.ndarray:
    #     data = data[
    #         np.logical_and(
    #             data[self.combo_x.currentText()] >= self.chart.xaxis.min(),
    #             data[self.combo_x.currentText()] <= self.chart.xaxis.max(),
    #         )
    #     ]
    #     data = data[
    #         np.logical_and(
    #             data[self.combo_y.currentText()] >= self.chart.yaxis.min(),
    #             data[self.combo_y.currentText()] <= self.chart.yaxis.max(),
    #         )
    #     ]
    #     return data


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
        "largest_feret_diameters_µm": "radius",
    }

    def __init__(self, path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.resize(1200, 800)

        self.data = np.genfromtxt(path, names=True, delimiter=",")
        assert self.data.dtype.names is not None

        # fixes for specific inputs
        if "particle_diameter_aspect_ratios_1" in self.data.dtype.names:
            self.data["particle_diameter_aspect_ratios_1"] = (
                1.0 / self.data["particle_diameter_aspect_ratios_1"]
            )
        if "radius" in self.data.dtype.names:  # radius in pixels
            self.data["radius"] *= 2.0 * 0.46

        # convert to simple form
        self.data = rfn.rename_fields(self.data, ExplorerWindow.REMAP_NAMES)
        assert self.data.dtype.names is not None

        self.scatter = ScatterWidget(self.data)
        self.scatter.chart.roi.sigRegionChangeFinished.connect(self.redrawCapillary)
        self.scatter.chart.roi.sigRegionChangeFinished.connect(self.redrawHistogram)
        self.scatter.updateRequested.connect(self.redrawScatter)
        self.scatter.updateRequested.connect(self.redrawCapillary)

        self.hist = HistogramChart()
        self.hist.setLimits(
            xMin=0.0, xMax=self.data["radius"].max(), yMin=0.0, yMax=100.0
        )
        self.hist.region.setRegion(np.percentile(self.data["radius"], [1, 99]))
        self.hist.region.setBounds((0.0, self.data["radius"].max()))
        self.hist.region.sigRegionChangeFinished.connect(self.redrawCapillary)
        self.hist.region.sigRegionChangeFinished.connect(self.redrawScatter)

        self.hist.cursorMoved.connect(self.printCursorPos)

        self.view_cap = QtWidgets.QGraphicsView()
        self.view_cap.setScene(QtWidgets.QGraphicsScene())

        self.image_cap = ImageItem(
            QtGui.QImage(),
            QtCore.QRectF(
                0.0, 0.0, ExplorerWindow.CAMERA_SIZE[0], ExplorerWindow.CAMERA_SIZE[1]
            ),
        )
        self.view_cap.scene().addItem(self.image_cap)
        self.view_cap.setSceneRect(self.image_cap.boundingRect())
        self.view_cap.fitInView(
            self.image_cap.boundingRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        )
        self.view_cap.scale(0.5, 0.5)

        self.label_count = QtWidgets.QLabel()

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
        controls_layout.addRow(self.label_count)
        for name, slider in self.sliders.items():
            controls_layout.addRow(name.replace("_", " ").title(), slider)

        controls_widget = QtWidgets.QWidget()
        controls_widget.setMinimumWidth(300)
        controls_widget.setLayout(controls_layout)

        capillary_dock = QtWidgets.QDockWidget("Capillary")
        capillary_dock.setWidget(self.view_cap)
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
            [controls_dock, hist_dock], [400, 800], QtCore.Qt.Orientation.Horizontal
        )

        self.redrawAll()

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
                np.logical_and(data["radius"] >= hist_min, data["radius"] <= hist_max)
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
        self.updateCanvasCapillary(data)

    def redrawHistogram(self):
        data = self.filteredData(scatter=True)
        self.hist.updateHistogram(data["radius"], bins=100)

    def redrawScatter(self):
        data = self.filteredData(hist=True)
        self.scatter.updateScatter(data)

    def updateCanvasCapillary(self, data: np.ndarray):
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

        self.image_cap.setImage(image)
