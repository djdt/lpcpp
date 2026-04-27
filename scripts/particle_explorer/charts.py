from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np

from particle_explorer.colors import cividis

import pyqtgraph
#
# class BaseChart(QtCharts.QChartView):
#     cursorPositionChanged = QtCore.Signal(float, float)
#
#     """QChartView with basic mouse naviagtion and styling.
#
#     Valid keys for 'theme' are "background", "axis", "grid", "title", "text".
#     A context menu implements copying the chart to the system clipboard and reseting the chart view.
#
#     Args:
#         chart: chart to display
#         theme: dict of theme keys and colors
#         allow_navigation: use mouse navigation
#         parent: parent widget
#     """
#
#     def __init__(
#         self,
#         chart: QtCharts.QChart,
#         parent: QtWidgets.QWidget | None = None,
#     ):
#         # chart.setBackgroundBrush(QtGui.QBrush(self.theme["background"]))
#         # chart.setBackgroundPen(QtGui.QPen(self.theme["background"]))
#         super().__init__(chart, parent)
#
#         self.xaxis = QtCharts.QValueAxis()
#         self.yaxis = QtCharts.QValueAxis()
#
#         self.chart().legend().setVisible(False)
#
#         self.limits = (0.0, 1.0, 0.0, 1.0)
#
#         self.chart().addAxis(self.xaxis, QtCore.Qt.AlignmentFlag.AlignBottom)
#         self.chart().addAxis(self.yaxis, QtCore.Qt.AlignmentFlag.AlignLeft)
#
#     def setRange(
#         self,
#         xmin: float | None = None,
#         xmax: float | None = None,
#         ymin: float | None = None,
#         ymax: float | None = None,
#     ):
#         if xmin is not None or xmax is not None:
#             self.xaxis.setRange(xmin or self.xaxis.min(), xmax or self.xaxis.max())
#         if ymin is not None or ymax is not None:
#             self.yaxis.setRange(ymin or self.yaxis.min(), ymax or self.yaxis.max())
#         self.chart().update()
#
#     def setLimits(
#         self,
#         xmin: float | None = None,
#         xmax: float | None = None,
#         ymin: float | None = None,
#         ymax: float | None = None,
#     ):
#         self.limits = (
#             xmin or self.limits[0],
#             xmax or self.limits[1],
#             ymin or self.limits[2],
#             ymax or self.limits[3],
#         )
#
#     def mouseMoveEvent(self, event: QtGui.QMouseEvent):
#         super().mouseMoveEvent(event)
#         pos = self.chart().mapToValue(event.position())
#         self.cursorPositionChanged.emit(pos.x(), pos.y())
#
#     def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
#         action_copy_image = QtGui.QAction(
#             QtGui.QIcon.fromTheme("insert-image"), "Copy To Clipboard", self
#         )
#         action_copy_image.setStatusTip("Copy the graphics view to the clipboard.")
#         action_copy_image.triggered.connect(self.copyToClipboard)
#
#         action_reset_zoom = QtGui.QAction(
#             QtGui.QIcon.fromTheme("zoom-original"), "Reset Zoom", self
#         )
#         action_reset_zoom.setStatusTip("Reset the chart to the orignal view.")
#         action_reset_zoom.triggered.connect(self.chart().zoomReset)
#
#         menu = QtWidgets.QMenu(self)
#         menu.addAction(action_copy_image)
#         menu.addAction(action_reset_zoom)
#         menu.popup(event.globalPos())
#
#     def copyToClipboard(self) -> None:
#         """Copy image of the current chart to the system clipboard."""
#         QtWidgets.QApplication.clipboard().setPixmap(self.grab(self.viewport().rect()))


class BaseChart(pyqtgraph.PlotWidget):
    cursorMoved = QtCore.Signal(QtCore.QPointF)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent, background="white")

        self.xaxis = pyqtgraph.AxisItem("bottom")
        # self.xaxis.setLabel(xlabel, units=xunits)

        self.yaxis = pyqtgraph.AxisItem("left")
        # self.yaxis.setLabel(ylabel, units=yunits)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):  # type: ignore
        self.cursorMoved.emit(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)


class HistogramChart(BaseChart):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        self.enableAutoRange(y=True)

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

    def updateScatter(self, xs: np.ndarray, ys: np.ndarray):
        self.series.setData(x=xs, y=ys)
        self.setLimits(xMin=xs.min(), xMax=xs.max(), yMin=ys.min(), yMax=ys.max())
