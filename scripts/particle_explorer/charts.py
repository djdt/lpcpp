from PySide6 import QtCore, QtGui, QtWidgets
from PySide6 import QtCharts

import numpy as np

from particle_explorer.colors import cividis

#
# class NiceValueAxis(QtCharts.QValueAxis):
#     """A chart axis that uses easy to read tick intervals.
#
#     Uses *at least* 'nticks' ticks, may be more.
#     """
#
#     nicenums = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5]
#
#     def __init__(self, nticks: int = 6, parent: QtCore.QObject | None = None):
#         super().__init__(parent)
#         self.nticks = nticks
#
#         self.setLabelFormat("%.4g")
#         self.setTickType(QtCharts.QValueAxis.TickType.TicksDynamic)
#         self.setTickAnchor(0.0)
#         self.setTickInterval(1e3)
#
#     def setRange(self, amin: float, amax: float) -> None:
#         self.fixValues(amin, amax)
#         super().setRange(amin, amax)
#
#     def fixValues(self, amin: float, amax: float) -> None:
#         delta = amax - amin
#
#         interval = delta / self.nticks
#         pwr = 10 ** int(np.log10(interval) - (1 if interval < 1.0 else 0))
#         interval = interval / pwr
#
#         idx = np.searchsorted(NiceValueAxis.nicenums, interval)
#         idx = min(idx, len(NiceValueAxis.nicenums) - 1)
#
#         interval = NiceValueAxis.nicenums[idx] * pwr
#         anchor = int(amin / interval) * interval
#
#         self.setTickAnchor(anchor)
#         self.setTickInterval(interval)


class ValueAxis(QtCharts.QValueAxis):
    def __init__(
        self, limits: tuple[float, float], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)

        self.limits = limits

        self.rangeChanged.connect(self.limitRange)

    def limitRange(self):
        if self.min() < self.limits[0]:
            self.setMin(self.limits[0])
        if self.max() > self.limits[1]:
            self.setMax(self.limits[1])


class BaseChart(QtCharts.QChartView):
    cursorPositionChanged = QtCore.Signal(float, float)

    """QChartView with basic mouse naviagtion and styling.

    Valid keys for 'theme' are "background", "axis", "grid", "title", "text".
    A context menu implements copying the chart to the system clipboard and reseting the chart view.

    Args:
        chart: chart to display
        theme: dict of theme keys and colors
        allow_navigation: use mouse navigation
        parent: parent widget
    """

    def __init__(
        self,
        chart: QtCharts.QChart,
        parent: QtWidgets.QWidget | None = None,
    ):
        # chart.setBackgroundBrush(QtGui.QBrush(self.theme["background"]))
        # chart.setBackgroundPen(QtGui.QPen(self.theme["background"]))
        super().__init__(chart, parent)

        self.xaxis = QtCharts.QValueAxis()
        self.yaxis = QtCharts.QValueAxis()

        self.chart().legend().setVisible(False)

        self.limits = (0.0, 1.0, 0.0, 1.0)

        self.chart().addAxis(self.xaxis, QtCore.Qt.AlignmentFlag.AlignBottom)
        self.chart().addAxis(self.yaxis, QtCore.Qt.AlignmentFlag.AlignLeft)

    def setRange(
        self,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
    ):
        if xmin is not None or xmax is not None:
            self.xaxis.setRange(xmin or self.xaxis.min(), xmax or self.xaxis.max())
        if ymin is not None or ymax is not None:
            self.yaxis.setRange(ymin or self.yaxis.min(), ymax or self.yaxis.max())
        self.chart().update()

    def setLimits(
        self,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
    ):
        self.limits = (
            xmin or self.limits[0],
            xmax or self.limits[1],
            ymin or self.limits[2],
            ymax or self.limits[3],
        )

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        super().mouseMoveEvent(event)
        pos = self.chart().mapToValue(event.position())
        self.cursorPositionChanged.emit(pos.x(), pos.y())

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        action_copy_image = QtGui.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy To Clipboard", self
        )
        action_copy_image.setStatusTip("Copy the graphics view to the clipboard.")
        action_copy_image.triggered.connect(self.copyToClipboard)

        action_reset_zoom = QtGui.QAction(
            QtGui.QIcon.fromTheme("zoom-original"), "Reset Zoom", self
        )
        action_reset_zoom.setStatusTip("Reset the chart to the orignal view.")
        action_reset_zoom.triggered.connect(self.chart().zoomReset)

        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_image)
        menu.addAction(action_reset_zoom)
        menu.popup(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy image of the current chart to the system clipboard."""
        QtWidgets.QApplication.clipboard().setPixmap(self.grab(self.viewport().rect()))


class HistogramChart(BaseChart):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(QtCharts.QChart(), parent=parent)

        self.setRubberBand(QtCharts.QChartView.RubberBand.HorizontalRubberBand)

        self._top_series = QtCharts.QLineSeries()
        self._bottom_series = QtCharts.QLineSeries()

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1)
        pen.setCosmetic(True)

        self.series = QtCharts.QAreaSeries(self._top_series, self._bottom_series)
        self.series.setPen(pen)
        self.series.setBrush(QtGui.QBrush(cividis[64]))

        self.chart().addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

    def updateHistogram(
        self,
        data: np.ndarray,
        bins: np.ndarray | int | None = None,
        density: bool = False,
        scale_yaxis: bool = True,
    ):
        counts, edges = np.histogram(data, bins=bins, density=density)  # type: ignore

        if scale_yaxis:
            self.setLimits(ymax=counts.max() * 1.05)
            self.setRange(ymax=counts.max() * 1.05)

        xs = np.repeat(edges.astype(np.float64), 2)
        ys = np.concatenate(([0.0], np.repeat(counts, 2), [0.0])).astype(np.float64)

        self._top_series.replaceNp(xs, ys)  # type: ignore


class ScatterChart(BaseChart):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(QtCharts.QChart(), parent=parent)

        self.setRubberBand(QtCharts.QChartView.RubberBand.RectangleRubberBand)

        self.series = QtCharts.QScatterSeries()
        self.series.setColor(QtCore.Qt.GlobalColor.black)
        self.series.setSelectedColor(QtGui.QColor(QtCore.Qt.GlobalColor.red))
        self.series.setMarkerSize(3)
        self.series.setUseOpenGL(True)
        self.chart().addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

    def updateScatter(self, xs: np.ndarray, ys: np.ndarray):
        self.series.replaceNp(xs.astype(np.float64), ys.astype(np.float64))  # type: ignore

    # def mousePressEvent(self, event:QtGui.QMouseEvent):
    #     self.drag_start = self.chart().mapToValue(event.position())
    #     super().mousePressEvent(event)
    #
    # def mouseReleaseEvent(self, event:QtGui.QMouseEvent):
    #     drag_end = self.chart().mapToValue(event.position())
    #
    #     rect = QtCore.QRectF(self.drag_start, drag_end)
    #
    #     for i, point in enumerate(self.series.points()):
    #         if rect.contains(point):
    #             self.series.selectPoint(i)
    #     #         selected.append(i)
    #     #
    #     # self.series.selectPoints(selected)
    #     self.update()
    #     print(self.series.selectedPoints())
    #
    #     super().mouseReleaseEvent(event)
