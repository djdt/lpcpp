import numpy as np

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from particle_explorer.colors import cividis
from particle_explorer.charts import HistogramChart
from particle_explorer.widgets import LabeledRangeSlider


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


class ExplorerWindow(QtWidgets.QMainWindow):
    CAMERA_SIZE = 2048, 1536

    def __init__(self, path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.resize(800, 600)

        self.data = np.genfromtxt(path, names=True, delimiter=",")

        self.chart_hist = HistogramChart()
        self.chart_hist.xaxis.setRange(0.0, self.data["radius"].max() * 2.0 * 0.46)
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
        self.view_cap.fitInView(self.image_cap.boundingRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding)

        self.aspect = LabeledRangeSlider(scale=100)
        self.aspect.setRange(0.0, 1.0)
        self.aspect.setValues(0.0, 1.0)

        self.convexity = LabeledRangeSlider(scale=100)
        self.convexity.setRange(0, 1)
        self.convexity.setValues(0, 1)

        self.circularity = LabeledRangeSlider(scale=100)
        self.circularity.setRange(0, 1)
        self.circularity.setValues(0, 1)

        self.frame_count = LabeledRangeSlider()
        self.frame_count.setRange(1, np.amax(self.data["frame_count"]))
        self.frame_count.setValues(1, np.amax(self.data["frame_count"]))

        self.aspect.rangeChanged.connect(self.redraw)
        self.convexity.rangeChanged.connect(self.redraw)
        self.circularity.rangeChanged.connect(self.redraw)
        self.frame_count.rangeChanged.connect(self.redraw)

        controls_layout = QtWidgets.QFormLayout()
        controls_layout.addRow("Aspect", self.aspect)
        controls_layout.addRow("Convex.", self.convexity)
        controls_layout.addRow("Circ.", self.circularity)
        controls_layout.addRow("Frame #", self.frame_count)

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
        self.redraw()

    def redraw(self):
        data = self.data[
            np.logical_and(
                self.data["aspect"] >= self.aspect.min(),
                self.data["aspect"] <= self.aspect.max(),
            )
        ]
        data = data[
            np.logical_and(
                data["convexity"] >= self.convexity.min(),
                data["convexity"] <= self.convexity.max(),
            )
        ]
        data = data[
            np.logical_and(
                data["circularity"] >= self.circularity.min(),
                data["circularity"] <= self.circularity.max(),
            )
        ]
        data = data[
            np.logical_and(
                data["frame_count"] >= self.frame_count.min(),
                data["frame_count"] <= self.frame_count.max(),
            )
        ]

        hist_range = 0.0, self.data["radius"].max()

        self.updateCanvasCapillary(data)
        self.updateCanvasHistogram(data, hist_range)

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
