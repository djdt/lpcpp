import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from ilmex.colors import cividis


class BaseChart(pyqtgraph.PlotWidget):
    cursorMoved = QtCore.Signal(QtCore.QPointF)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent, background="white")

        self.xaxis = pyqtgraph.AxisItem("bottom")
        self.yaxis = pyqtgraph.AxisItem("left")

        self.getPlotItem().setAxisItems({"bottom": self.xaxis, "left": self.yaxis})

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):  # type: ignore
        super().mouseMoveEvent(event)
        if self.plotItem is None:
            return
        self.cursorMoved.emit(self.plotItem.mapToView(event.position()))


class HistogramChart(BaseChart):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        self.enableAutoRange(y=True)
        self.xaxis.setLabel("Size (µm)")
        self.yaxis.setLabel("Count")

        self.series_filtered = pyqtgraph.PlotCurveItem(
            x=[0, 0],
            y=[0],
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            brush=QtGui.QBrush(QtCore.Qt.GlobalColor.lightGray),
            skipFiniteCheck=True,
        )
        self.addItem(self.series_filtered)
        self.series = pyqtgraph.PlotCurveItem(
            x=[0, 0],
            y=[0],
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            brush=QtGui.QBrush(cividis[64]),
            skipFiniteCheck=True,
        )
        self.addItem(self.series)

        self.region = pyqtgraph.LinearRegionItem(
            (0.0, 1.0),
            pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 0),
            brush=QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush),
            hoverBrush=QtGui.QBrush(QtGui.QColor(255, 0, 0, 32)),
        )
        self.addItem(self.region)

    def updateHistogram(
        self,
        data: np.ndarray,
        filter: np.ndarray | None = None,
        bins: np.ndarray | int | None = None,
        density: bool = False,
    ):
        edges = np.histogram_bin_edges(data, bins=bins)
        if filter is not None:
            counts, _ = np.histogram(data[filter], bins=edges, density=density)
            self.series.setData(y=counts, x=edges)
            counts, _ = np.histogram(data, bins=edges, density=density)
            self.series_filtered.setData(y=counts, x=edges)
        else:
            counts, _ = np.histogram(data, bins=edges, density=density)
            self.series.setData(y=counts, x=edges)
            self.series_filtered.clear()

        self.setLimits(yMax=counts.max() * 1.1)
        self.autoRange()


class ScatterChart(BaseChart):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.series = pyqtgraph.ScatterPlotItem(
            size=5, symbol="o", pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 0)
        )
        self.addItem(self.series)

        self.series_filtered = pyqtgraph.ScatterPlotItem(
            size=5, symbol="o", pen=QtGui.QPen(QtCore.Qt.GlobalColor.lightGray, 0)
        )
        self.addItem(self.series_filtered)

        self.roi = pyqtgraph.RectROI(
            (0.0, 0.0),
            (0.0, 0.0),
            pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 0),
            handlePen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 0),
            hoverPen=QtGui.QPen(QtCore.Qt.GlobalColor.red, 0),
            handleHoverPen=QtGui.QPen(QtCore.Qt.GlobalColor.red, 0),
        )
        self.roi.addScaleHandle((0, 1), (1, 0))
        self.roi.addScaleHandle((1, 0), (0, 1))
        self.roi.addScaleHandle((0, 0), (1, 1))
        self.addItem(self.roi)

    def updateScatter(
        self, xs: np.ndarray, ys: np.ndarray, mask: np.ndarray | None = None
    ):
        if mask is not None:
            self.series.setData(x=xs[mask], y=ys[mask])
            self.series_filtered.setData(x=xs[~mask], y=ys[~mask])
        else:
            self.series.setData(x=xs, y=ys)
            self.series_filtered.clear()
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        dx, dy = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
        self.setLimits(xMin=xmin - dx, xMax=xmax + dx, yMin=ymin - dy, yMax=ymax + dy)
        self.getPlotItem().setXRange(xmin - dx, xmax + dx)
        self.getPlotItem().setYRange(ymin - dy, ymax + dy)


class TimeSeriesChart(BaseChart):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.xaxis.setLabel("Frame")
        self.yaxis.setLabel("Count")

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
        pen.setCosmetic(True)
        self.series_mean = pyqtgraph.PlotCurveItem(
            x=[0], y=[0], pen=pen, skipFiniteCheck=True
        )
        self.addItem(self.series_mean)

        self.series_std_upper = pyqtgraph.PlotCurveItem(
            x=[0], y=[0], brush=QtGui.QBrush(cividis[128]), skipFiniteCheck=True
        )
        # self.addItem(self.series_std_upper)
        self.series_std_lower = pyqtgraph.PlotCurveItem(
            x=[0], y=[0], brush=QtGui.QBrush(cividis[128]), skipFiniteCheck=True
        )
        # self.addItem(self.series_std_lower)

        self.series_std = pyqtgraph.FillBetweenItem(
            self.series_std_lower,
            self.series_std_upper,
            brush=QtGui.QBrush(cividis[240]),
        )
        self.addItem(self.series_std)

    def updateTimeSeries(self, data: np.ndarray):
        if data.size == 0:
            self.series_mean.setData(x=[], y=[])
            return

        bins = np.linspace(data["frame"].min(), data["frame"].max(), 100)[:-1]
        idx = np.digitize(data["frame"], bins)
        counts = np.bincount(idx)[1:]

        view = np.lib.stride_tricks.sliding_window_view(counts, 5)
        mean = np.mean(view, axis=-1)
        std = np.std(view, axis=-1)

        self.series_mean.setData(x=bins[2:-2], y=mean)
        self.series_std_upper.setData(x=bins[2:-2], y=mean + std)
        self.series_std_lower.setData(x=bins[2:-2], y=mean - std)
        self.setLimits(
            xMin=bins[0] - 1, xMax=bins[-1] + 1, yMin=0, yMax=np.amax(counts) * 1.1
        )
        self.autoRange()
