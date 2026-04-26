from PySide6 import QtCore, QtGui, QtWidgets


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
