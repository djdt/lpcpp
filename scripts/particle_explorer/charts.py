from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np

from particle_explorer.colors import cividis

import pyqtgraph


class BaseChart(pyqtgraph.PlotWidget):
    cursorMoved = QtCore.Signal(QtCore.QPointF)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent, background="white")

        self.xaxis = pyqtgraph.AxisItem("bottom")
        self.yaxis = pyqtgraph.AxisItem("left")

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

        brush = QtGui.QBrush(cividis[64])
        self.series = pyqtgraph.PlotCurveItem(
            x=[0, 0],
            y=[0],
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            brush=brush,
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
        bins: np.ndarray | int | None = None,
        density: bool = False,
        scale_yaxis: bool = True,
    ):
        counts, edges = np.histogram(data, bins=bins, density=density)  # type: ignore

        self.series.setData(y=counts, x=edges)
        self.setLimits(yMax=counts.max() * 1.1)
        self.autoRange()


class ScatterChart(BaseChart):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.series = pyqtgraph.ScatterPlotItem(
            size=5, symbol="o", pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 0)
        )
        self.addItem(self.series)

        self.roi = pyqtgraph.RectROI(
            (0.0, 0.0),
            (0.0, 0.0),
            pen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 0),
            handlePen=QtGui.QPen(QtCore.Qt.GlobalColor.black, 0),
            hoverPen=QtGui.QPen(QtCore.Qt.GlobalColor.red, 0),
            handleHoverPen=QtGui.QPen(QtCore.Qt.GlobalColor.red, 0),
            sideScalers=True,
        )
        self.addItem(self.roi)

    def updateScatter(self, xs: np.ndarray, ys: np.ndarray):
        self.series.setData(x=xs, y=ys)
        self.setLimits(xMin=xs.min(), xMax=xs.max(), yMin=ys.min(), yMax=ys.max())

        xmin, xmax = np.percentile(xs, [1, 99])
        ymin, ymax = np.percentile(ys, [1, 99])
        self.roi.setPos(xmin, ymin)
        self.roi.setSize(QtCore.QPointF(xmax - xmin, ymax - ymin))
