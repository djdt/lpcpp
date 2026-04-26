from PySide6 import QtCore, QtGui, QtWidgets
from PySide6 import QtCharts

import numpy as np

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
        allow_navigation: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.allow_navigation = allow_navigation

        self.nav_pos: QtCore.QPointF | None = None

        # chart.setBackgroundBrush(QtGui.QBrush(self.theme["background"]))
        # chart.setBackgroundPen(QtGui.QPen(self.theme["background"]))
        super().__init__(chart, parent)

        self.xaxis = QtCharts.QValueAxis()
        self.yaxis = QtCharts.QValueAxis()

        self.limits = (0.0, 1.0, 0.0, 1.0)

        self.chart().addAxis(self.xaxis, QtCore.Qt.AlignmentFlag.AlignBottom)
        self.chart().addAxis(self.yaxis, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.xaxis.rangeChanged.connect(self.onRangeChanged)
        self.yaxis.rangeChanged.connect(self.onRangeChanged)

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

    def onRangeChanged(self):
        return
        if self.xaxis.min() < self.limits[0]:
            self.xaxis.setMin(self.limits[0])
        if self.xaxis.max() > self.limits[1]:
            self.xaxis.setMax(self.limits[1])

        if self.yaxis.min() < self.limits[2]:
            self.yaxis.setMin(self.limits[2])
        if self.yaxis.max() > self.limits[3]:
            self.yaxis.setMax(self.limits[3])

    # def addAxis(
    #     self, axis: QtCharts.QAbstractAxis, alignment: QtCore.Qt.AlignmentFlag
    # ) -> None:
    #     axis.setMinorGridLineVisible(True)
    #     # axis.setTitleBrush(QtGui.QBrush(self.theme["title"]))
    #     # axis.setGridLinePen(QtGui.QPen(self.theme["grid"], 1.0))
    #     # axis.setMinorGridLinePen(QtGui.QPen(self.theme["grid"], 0.5))
    #     # axis.setLinePen(QtGui.QPen(self.theme["axis"], 1.0))
    #     # axis.setLabelsColor(self.theme["text"])
    #     # axis.setShadesColor(self.theme["title"])
    #
    #     if isinstance(axis, QtCharts.QValueAxis):
    #         axis.setTickCount(6)
    #     self.chart().addAxis(axis, alignment)

    # def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
    #     if (
    #         event.button() == QtCore.Qt.MouseButton.MiddleButton
    #         and self.allow_navigation
    #     ):
    #         self.nav_pos = event.position()
    #         self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
    #     else:
    #         super().mousePressEvent(event)
    #
    # def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
    #     if self.nav_pos is not None and self.allow_navigation:
    #         pos = event.position()
    #         offset = self.nav_pos - pos
    #         if (
    #             offset.x() + self.xaxis.min() < self.limits[0]
    #             or offset.x() + self.xaxis.max() > self.limits[1]
    #         ):
    #             offset.setX(0.0)
    #         if (
    #             -offset.y() + self.yaxis.min() < self.limits[2]
    #             or -offset.y() + self.yaxis.max() > self.limits[3]
    #         ):
    #             offset.setY(0.0)
    #         self.chart().scroll(offset.x(), -offset.y())
    #         self.nav_pos = event.position()
    #     else:
    #         super().mouseMoveEvent(event)
    #
    # def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
    #     self.nav_pos = None
    #     self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
    #     super().mouseReleaseEvent(event)
    #
    # def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
    #     if self.allow_navigation:
    #         pos = self.chart().mapToValue(event.position())
    #         scale = pow(2, event.angleDelta().y() / 360.0)
    #
    #         x0, x1 = self.xaxis.min(), self.xaxis.max()
    #         y0, y1 = self.yaxis.min(), self.yaxis.max()
    #
    #         w, h = (x1 - x0), (y1 - y0)
    #         nw, nh = w * scale, h * scale
    #
    #         dx, dy = nw - w, nh - h
    #
    #         h = (y1 - y0) * scale
    #
    #         self.xaxis.setMin(x0 + dx * (pos.x() - x0) / w)
    #         self.xaxis.setMax(x1 - dx * (x1 - pos.x()) / w)
    #         self.yaxis.setMin(y0 + dy * (pos.x() - y0) / h)
    #         self.yaxis.setMax(y1 + dy * (y1 - pos.y()) / h)
    #
    #     super().wheelEvent(event)
    #
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
        super().__init__(QtCharts.QChart(), allow_navigation=False, parent=parent)

        self.setRubberBand(QtCharts.QChartView.RubberBand.HorizontalRubberBand)

        self._top_series = QtCharts.QLineSeries()
        self._bottom_series = QtCharts.QLineSeries()

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1)
        pen.setCosmetic(True)

        self.histogram_series = QtCharts.QAreaSeries(
            self._top_series, self._bottom_series
        )
        self.histogram_series.setPen(pen)
        self.histogram_series.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.green))

        self.chart().addSeries(self.histogram_series)
        self.histogram_series.attachAxis(self.xaxis)
        self.histogram_series.attachAxis(self.yaxis)

    def updateHistogram(
        self,
        data: np.ndarray,
        bins: np.ndarray | int | None = None,
        density: bool = False,
        scale_yaxis: bool = True,
    ):
        counts, edges = np.histogram(data, bins=bins, density=density)

        if scale_yaxis:
            self.setLimits(ymax=counts.max() * 1.05)
            self.setRange(ymax=counts.max() * 1.05)

        xs = np.repeat(edges.astype(np.float64), 2)
        ys = np.concatenate(([0.0], np.repeat(counts, 2), [0.0])).astype(np.float64)

        self._top_series.replaceNp(xs, ys)
