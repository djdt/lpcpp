import numpy as np

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from particle_explorer.colors import cividis
from particle_explorer.charts import HistogramChart
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


class ExplorerWindow(QtWidgets.QMainWindow):
    CAMERA_SIZE = 2048, 1536
    VALID_RANGES = {  # name : (min val, max val, scale)
        "frame_count": (1, None, 1),
        "area": (0, None, 1),
        "aspect": (0, 1.0, 1e-2),
        "circularity": (0, 1.0, 1e-2),
        "convexity": (0, 1.0, 1e-2),
        "intensity": (0, None, 1),
        "x": (None, None, 1),
        "y": (None, None, 1),
    }

    def __init__(self, path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.resize(1200, 800)

        self.data = np.genfromtxt(path, names=True, delimiter=",")

        self.chart_hist = HistogramChart()
        self.chart_hist.setLimits(
            0.0, self.data["radius"].max() * 2.0 * 0.46, 0.0, 100.0
        )
        self.chart_hist.setRange(0.0, self.data["radius"].max() * 2.0 * 0.46)
        self.chart_hist.xaxis.rangeChanged.connect(self.redraw)

        self.chart_hist.xaxis.setTitleText("Size (µm)")

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

        self.status_bar = self.statusBar()

        self.sliders = {}
        for name, (vmin, vmax, scale) in ExplorerWindow.VALID_RANGES.items():
            if vmin is None:
                vmin = self.data[name].min()
            if vmax is None:
                vmax = self.data[name].max()

            self.sliders[name] = ControlSlider(name, scale=scale)
            self.sliders[name].setRangeScaled(vmin, vmax)
            self.sliders[name].setScaled(vmin, vmax)
            self.sliders[name].rangeChanged.connect(self.redraw)
            self.sliders[name].rangeChanged.connect(self.printControl)

        controls_layout = QtWidgets.QFormLayout()
        for name, slider in self.sliders.items():
            controls_layout.addRow(name.replace("_", " ").title(), slider)

        controls_widget = QtWidgets.QWidget()
        controls_widget.setMinimumWidth(300)
        controls_widget.setLayout(controls_layout)

        capillary_dock = QtWidgets.QDockWidget()
        capillary_dock.setWidget(self.view_cap)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, capillary_dock)

        controls_dock = QtWidgets.QDockWidget()
        controls_dock.setWidget(controls_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, controls_dock)

        self.setCentralWidget(self.chart_hist)
        # self.setStatusBar(self.status_bar)

        self.redraw()

    def printControl(self, name: str, min: float, max: float):
        self.status_bar.showMessage(f"{name}: {min:.12g} - {max:.12g}")

    def redraw(self):
        data = self.data
        for name, slider in self.sliders.items():
            vmin, vmax = slider.scaled()
            data = data[np.logical_and(data[name] >= vmin, data[name] <= vmax)]

        hist_range = 0.0, self.data["radius"].max()

        self.updateCanvasHistogram(data, hist_range)
        data = data[
            np.logical_and(
                data["radius"] >= self.chart_hist.xaxis.min(),
                data["radius"] <= self.chart_hist.xaxis.max(),
            )
        ]
        self.updateCanvasCapillary(data)

    def updateCanvasHistogram(self, data: np.ndarray, xlim: tuple[float, float]):
        if data.size == 0:
            return

        self.chart_hist.updateHistogram(data["radius"] * 2.0 * 0.46, bins=100)

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
