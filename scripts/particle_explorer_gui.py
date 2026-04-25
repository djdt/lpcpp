import numpy as np

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6 import QtCore, QtGui, QtWidgets

import pyqtgraph

CAMERA_SIZE = 1536, 2048

ctable_cividis = [
    4278198861,
    4278199119,
    4278199120,
    4278199378,
    4278199636,
    4278199893,
    4278199895,
    4278200153,
    4278200411,
    4278200412,
    4278200670,
    4278200928,
    4278200930,
    4278201188,
    4278201446,
    4278201447,
    4278201705,
    4278201963,
    4278202221,
    4278202223,
    4278202480,
    4278202480,
    4278202736,
    4278202736,
    4278465136,
    4278727536,
    4278924144,
    4279121008,
    4279317871,
    4279514735,
    4279645807,
    4279777135,
    4279908463,
    4280039534,
    4280105326,
    4280236654,
    4280367982,
    4280433518,
    4280564846,
    4280630637,
    4280761709,
    4280827501,
    4280958829,
    4281024365,
    4281090157,
    4281221484,
    4281287276,
    4281352812,
    4281418604,
    4281484396,
    4281615468,
    4281681260,
    4281747052,
    4281812588,
    4281878380,
    4281944172,
    4282009707,
    4282075499,
    4282206827,
    4282272619,
    4282338155,
    4282403947,
    4282469739,
    4282535275,
    4282601067,
    4282666859,
    4282732395,
    4282798187,
    4282863979,
    4282929515,
    4282995307,
    4283061099,
    4283126892,
    4283192428,
    4283258220,
    4283324012,
    4283324012,
    4283389804,
    4283455596,
    4283521132,
    4283586924,
    4283652716,
    4283718252,
    4283784045,
    4283849837,
    4283915629,
    4283981165,
    4284046957,
    4284047213,
    4284112749,
    4284178542,
    4284244334,
    4284309870,
    4284375662,
    4284441454,
    4284507246,
    4284572783,
    4284573039,
    4284638831,
    4284704367,
    4284770159,
    4284835952,
    4284901744,
    4284967280,
    4285033072,
    4285033329,
    4285098865,
    4285164657,
    4285230449,
    4285295986,
    4285361778,
    4285427570,
    4285427827,
    4285493363,
    4285559155,
    4285624947,
    4285690740,
    4285756276,
    4285822069,
    4285822325,
    4285887861,
    4285953654,
    4286019446,
    4286085238,
    4286150775,
    4286151031,
    4286216823,
    4286282615,
    4286348152,
    4286413944,
    4286479736,
    4286545272,
    4286611064,
    4286676856,
    4286742648,
    4286808184,
    4286873976,
    4286939768,
    4286940024,
    4287005560,
    4287071352,
    4287137144,
    4287202936,
    4287268472,
    4287334264,
    4287400056,
    4287465848,
    4287531384,
    4287597175,
    4287662967,
    4287728759,
    4287794295,
    4287860087,
    4287925879,
    4287991671,
    4288057207,
    4288122998,
    4288188790,
    4288254582,
    4288320374,
    4288385910,
    4288451702,
    4288517493,
    4288583285,
    4288648821,
    4288714613,
    4288780404,
    4288846196,
    4288911988,
    4288977524,
    4289043315,
    4289109107,
    4289174899,
    4289240691,
    4289306226,
    4289372018,
    4289437810,
    4289503601,
    4289569393,
    4289634929,
    4289700720,
    4289766512,
    4289832304,
    4289898095,
    4289963631,
    4290029423,
    4290095214,
    4290161006,
    4290226797,
    4290292589,
    4290358125,
    4290423916,
    4290489708,
    4290555499,
    4290621291,
    4290687082,
    4290752618,
    4290883945,
    4290949737,
    4291015528,
    4291081320,
    4291147111,
    4291212647,
    4291278438,
    4291344229,
    4291410021,
    4291475812,
    4291541604,
    4291607395,
    4291673186,
    4291738722,
    4291804513,
    4291870304,
    4292001632,
    4292067423,
    4292133214,
    4292199006,
    4292264797,
    4292330332,
    4292396123,
    4292461914,
    4292527706,
    4292593497,
    4292659288,
    4292790615,
    4292856406,
    4292922197,
    4292987988,
    4293053523,
    4293119314,
    4293185105,
    4293250896,
    4293316687,
    4293448014,
    4293513805,
    4293579596,
    4293645387,
    4293711178,
    4293776968,
    4293842759,
    4293908550,
    4294039876,
    4294105667,
    4294171202,
    4294236992,
    4294302783,
    4294368573,
    4294499899,
    4294565690,
    4294631480,
    4294697270,
    4294828596,
    4294828851,
    4294829364,
    4294829622,
    4294829879,
]


class RangeSlider(QtWidgets.QSlider):
    """A QSlider with two inputs.

    The slider is highlighted between the two selected values.
    """

    value2Changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self._value2 = 99
        self._pressed = False

    def left(self) -> int:
        """The leftmost value."""
        return min(self.value(), self.value2())

    def setLeft(self, value: int) -> None:
        """Set the leftmost value."""
        if self.value() < self._value2:
            self.setValue(value)
        else:
            self.setValue2(value)

    def right(self) -> int:
        """The rightmost value."""
        return max(self.value(), self.value2())

    def setRight(self, value: int) -> None:
        """Set the rightmost value."""
        if self.value() > self._value2:
            self.setValue(value)
        else:
            self.setValue2(value)

    def values(self) -> tuple[int, int]:
        """Returns the values (left, right)."""
        return self.left(), self.right()

    def setValues(self, left: int, right: int) -> None:
        """Set both values."""
        self.setValue(left)
        self.setValue2(right)

    def value2(self) -> int:
        """Raw access to the second slider value."""
        return self._value2

    def setValue2(self, value: int) -> None:
        """Raw setting of the second slider value."""
        self._value2 = value
        self.value2Changed.emit(self._value2)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self,
        )
        # Handle groove is minus 1/2 the handle width each side
        pos = self.style().sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            self.value2(),
            groove.width() - handle.width(),
        )
        pos += handle.width() // 2

        handle.moveCenter(QtCore.QPoint(pos, handle.center().y()))
        handle = handle.marginsAdded(QtCore.QMargins(2, 2, 2, 2))
        if handle.contains(event.position().toPoint()):
            event.accept()
            self._pressed = True
            self.setSliderDown(True)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pressed:
            pos = event.position().toPoint()
            option = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderHandle,
                self,
            )
            value = self.style().sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                pos.x() - handle.width() // 2,
                groove.width() - handle.width(),
            )
            handle.moveCenter(pos)
            if self.hasTracking():
                handle = handle.marginsAdded(
                    QtCore.QMargins(
                        handle.width() * 3,
                        handle.width(),
                        handle.width() * 3,
                        handle.width(),
                    )
                )
                self.setValue2(value)
                self.repaint(handle)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pressed:
            pos = event.position().toPoint()
            self._pressed = False
            option = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderHandle,
                self,
            )
            value = self.style().sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                pos.x() - handle.width() // 2,
                groove.width() - handle.width(),
            )
            self.setSliderDown(False)
            self.setValue2(value)
            self.update()

        super().mouseReleaseEvent(event)

    def paintEvent(
        self,
        event: QtGui.QPaintEvent,
    ) -> None:
        painter = QtGui.QPainter(self)
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        option.activeSubControls = QtWidgets.QStyle.SubControl.SC_None

        if self.isSliderDown():
            option.state |= QtWidgets.QStyle.StateFlag.State_Sunken
            option.activeSubControls = QtWidgets.QStyle.SubControl.SC_ScrollBarSlider
        else:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
            option.activeSubControls = self.style().hitTestComplexControl(
                QtWidgets.QStyle.ComplexControl.CC_Slider, option, pos, self
            )

        groove = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
            self,
        )
        start = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.left(), groove.width()
        )
        end = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.right(), groove.width()
        )

        # Draw grooves
        option.subControls = QtWidgets.QStyle.SubControl.SC_SliderGroove

        option.sliderPosition = self.maximum() - self.minimum() - self.left()
        option.upsideDown = not option.upsideDown

        cliprect = QtCore.QRect(groove)
        cliprect.setRight(end)
        painter.setClipRegion(QtGui.QRegion(cliprect))

        self.style().drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_Slider, option, painter, self
        )

        option.upsideDown = not option.upsideDown
        option.sliderPosition = self.right()
        cliprect.setLeft(start)
        cliprect.setRight(groove.right())
        painter.setClipRegion(QtGui.QRegion(cliprect))

        self.style().drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_Slider, option, painter, self
        )

        painter.setClipRegion(QtGui.QRegion())
        painter.setClipping(False)

        # Draw handles
        option.subControls = QtWidgets.QStyle.SubControl.SC_SliderHandle

        option.sliderPosition = self.left()
        self.style().drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_Slider, option, painter, self
        )
        option.sliderPosition = self.right()
        self.style().drawComplexControl(
            QtWidgets.QStyle.ComplexControl.CC_Slider, option, painter, self
        )


class LabeledRangeSlider(QtWidgets.QWidget):
    rangeChanged = QtCore.Signal()

    def __init__(self, scale: float = 1, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.slider = RangeSlider()

        self.label = QtWidgets.QLabel("(0, 99)")
        self.label.setMinimumWidth(
            self.fontMetrics().tightBoundingRect("x(000., 000.)").width()
        )

        self.label_scale = scale

        self.slider.valueChanged.connect(self.updateLabel)
        self.slider.value2Changed.connect(self.updateLabel)

        self.slider.valueChanged.connect(self.rangeChanged)
        self.slider.value2Changed.connect(self.rangeChanged)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

    def updateLabel(self):
        v1, v2 = self.slider.values()
        self.label.setText(
            f"({v1 / self.label_scale:>4.3g}, {v2 / self.label_scale:>4.3g})"
        )

    def min(self) -> float:
        return self.slider.value() / self.label_scale

    def max(self) -> float:
        return self.slider.value2() / self.label_scale


class ExplorerWindow(QtWidgets.QMainWindow):
    def __init__(self, path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.resize(800, 600)

        self.data = np.genfromtxt(path, names=True, delimiter=",")

        self.plot_hist = pyqtgraph.PlotWidget(background="white")
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1)
        pen.setCosmetic(True)
        brush = QtGui.QBrush(QtCore.Qt.GlobalColor.lightGray)
        self.curve_hist = pyqtgraph.PlotCurveItem(
            x=[0, 0],
            y=[0],
            stepMode="center",
            pen=pen,
            brush=brush,
            fillLevel=0,
            fillOutline=True,
            skipFiniteCheck=True,
        )
        self.plot_hist.addItem(self.curve_hist)
        self.plot_hist.setXRange(0.0, self.data["radius"].max() * 2.0 * 0.46)

        self.plot_cap = pyqtgraph.ImageView()

        self.aspect = LabeledRangeSlider(scale=100)
        self.aspect.slider.setRange(0, 100)
        self.aspect.slider.setValues(0, 100)

        self.convexity = LabeledRangeSlider(scale=100)
        self.convexity.slider.setRange(0, 100)
        self.convexity.slider.setValues(0, 100)

        self.circularity = LabeledRangeSlider(scale=100)
        self.circularity.slider.setRange(0, 100)
        self.circularity.slider.setValues(0, 100)

        self.frame_count = LabeledRangeSlider()
        self.frame_count.slider.setRange(1, np.amax(self.data["frame_count"]))
        self.frame_count.slider.setValues(1, np.amax(self.data["frame_count"]))

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
        capillary_dock.setWidget(self.plot_cap)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, capillary_dock)

        controls_dock = QtWidgets.QDockWidget()
        controls_dock.setWidget(controls_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, controls_dock)

        self.setCentralWidget(self.plot_hist)
        self.redraw()

    def redraw(self):
        data = self.data[
            np.logical_and(
                self.data["aspect"] > self.aspect.min(),
                self.data["aspect"] < self.aspect.max(),
            )
        ]
        data = data[
            np.logical_and(
                data["convexity"] > self.convexity.min(),
                data["convexity"] < self.convexity.max(),
            )
        ]
        data = data[
            np.logical_and(
                data["circularity"] > self.circularity.min(),
                data["circularity"] < self.circularity.max(),
            )
        ]
        data = data[
            np.logical_and(
                data["frame_count"] > self.frame_count.min(),
                data["frame_count"] < self.frame_count.max(),
            )
        ]

        hist_range = 0.0, self.data["radius"].max()

        self.updateCanvasHistogram(data, hist_range)
        self.updateCanvasCapillary(data)

    def updateCanvasHistogram(self, data: np.ndarray, xlim: tuple[float, float]):
        if data.size == 0:
            return
        bins = np.arange(0, data["radius"].max() * 2.0 * 0.46 + 1, 0.1)

        # counts = np.digitize(np.sort(data["radius"]), bins)

        counts, edges = np.histogram(data["radius"] * 2.0 * 0.46, bins)

        # xs = np.repeat(bins, 2).astype(np.float64)
        # ys = np.repeat(counts, 2).astype(np.float64)

        self.curve_hist.setData(edges, counts)
        # self.plot_hist.set()
        # self.plot_hist.addItem()

        # self.axis_hist_x.setMin(xlim[0])
        # self.axis_hist_x.setMax(xlim[1])
        #
        # self.axis_hist_y.setMin(0.0)
        # self.axis_hist_y.setMax(counts.max() * 1.05)
        #
        # self.series_hist.upperSeries().replaceNp(xs[1:], ys[:-1])

    def updateCanvasCapillary(self, data: np.ndarray):
        return
        array, _, _ = np.histogram2d(
            data["y"],
            data["x"],
            bins=(
                np.arange(0, CAMERA_SIZE[0], 10),
                np.arange(0, CAMERA_SIZE[1], 10),
            ),
        )
        self.plot_cap.setImage(array, levels=256)
        # vmin, vmax = 0.0, np.percentile(array, 95)
        # array = np.clip(array, vmin, vmax)
        # if vmin != vmax:
        #     array = (array - vmin) / (vmax - vmin)
        # array = (array * 254).astype(np.uint8) + 1
        # image = QtGui.QImage(
        #     array.data,
        #     array.shape[1],
        #     array.shape[0],
        #     array.strides[0],
        #     QtGui.QImage.Format.Format_Indexed8,
        # )
        # image._array = array
        # image.setColorTable(ctable_cividis)
        #
        # self.view_cap.scene().addPixmap(QtGui.QPixmap.fromImage(image))
        #
        # self.view_cap.setSceneRect(image.rect())


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    app.setApplicationName("Particle Explorer")

    win = ExplorerWindow(Path("/home/tom/Downloads/octanol/15_32_37_particles.csv"))

    win.show()
    app.exec()


def array_to_image(array: np.ndarray) -> QtGui.QImage:
    """Converts a numpy array to a Qt image."""

    # array = np.ascontiguousarray(array)
    image = QtGui.QImage(
        array.data,
        array.shape[1],
        array.shape[0],
        array.strides[0],
        QtGui.QImage.Format.Format_Indexed8,
    )
    image._array = array
    return image
