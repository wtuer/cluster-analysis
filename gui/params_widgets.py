from abc import ABC, abstractmethod

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QHBoxLayout
)


class BaseParamsWidget(QWidget):
    params_changed = pyqtSignal(dict)

    def __init__(self, schema, parent=None):
        super().__init__(parent)
        self._schema = schema
        self._widgets = {}
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._build_widgets()

    def _build_widgets(self):
        for item in self._schema:
            name = item["name"]
            wtype = item["type"]

            if wtype == "int":
                w = QSpinBox()
                w.setMinimum(item["min"])
                w.setMaximum(item["max"])
                w.setValue(item["default"])
                w.setSingleStep(item.get("step", 1))
                w.valueChanged.connect(self._on_changed)
                self._layout.addRow(item["label_cn"], w)
                self._widgets[name] = w

            elif wtype == "float":
                if item.get("nullable"):
                    container = QWidget()
                    h = QHBoxLayout(container)
                    h.setContentsMargins(0, 0, 0, 0)
                    cb = QCheckBox(item.get("nullable_label", "自动"))
                    cb.setChecked(item["default"] is None)
                    w = QDoubleSpinBox()
                    w.setMinimum(item["min"])
                    w.setMaximum(item["max"])
                    w.setValue(item["min"] if item["default"] is None else item["default"])
                    w.setSingleStep(item.get("step", 0.1))
                    w.setDecimals(2)
                    w.setEnabled(item["default"] is not None)
                    cb.toggled.connect(lambda checked, sp=w: sp.setEnabled(not checked))
                    cb.toggled.connect(self._on_changed)
                    w.valueChanged.connect(self._on_changed)
                    h.addWidget(cb)
                    h.addWidget(w)
                    self._layout.addRow(item["label_cn"], container)
                    self._widgets[name] = w
                    self._widgets[name + "_nullable"] = cb
                else:
                    w = QDoubleSpinBox()
                    w.setMinimum(item["min"])
                    w.setMaximum(item["max"])
                    w.setValue(item["default"])
                    w.setSingleStep(item.get("step", 0.1))
                    w.setDecimals(2)
                    w.valueChanged.connect(self._on_changed)
                    self._layout.addRow(item["label_cn"], w)
                    self._widgets[name] = w

            elif wtype == "combo":
                w = QComboBox()
                labels = item.get("option_labels_cn", item["options"])
                for label in labels:
                    w.addItem(label)
                if item["default"] in item["options"]:
                    w.setCurrentIndex(item["options"].index(item["default"]))
                w.currentIndexChanged.connect(self._on_changed)
                self._layout.addRow(item["label_cn"], w)
                self._widgets[name] = w

    def _on_changed(self, _=None):
        self.params_changed.emit(self.get_params())

    def get_params(self):
        params = {}
        for item in self._schema:
            name = item["name"]
            wtype = item["type"]

            if wtype == "int":
                params[name] = self._widgets[name].value()
            elif wtype == "float":
                if item.get("nullable"):
                    cb = self._widgets[name + "_nullable"]
                    if cb.isChecked():
                        params[name] = None
                    else:
                        params[name] = self._widgets[name].value()
                else:
                    params[name] = self._widgets[name].value()
            elif wtype == "combo":
                w = self._widgets[name]
                params[name] = item["options"][w.currentIndex()]
        return params

    def set_defaults(self):
        for item in self._schema:
            name = item["name"]
            wtype = item["type"]
            if wtype == "int":
                w = self._widgets[name]
                w.blockSignals(True)
                w.setValue(item["default"])
                w.blockSignals(False)
            elif wtype == "float":
                if item.get("nullable"):
                    cb = self._widgets[name + "_nullable"]
                    cb.blockSignals(True)
                    cb.setChecked(item["default"] is None)
                    cb.blockSignals(False)
                    if item["default"] is not None:
                        w = self._widgets[name]
                        w.blockSignals(True)
                        w.setValue(item["default"])
                        w.blockSignals(False)
                else:
                    w = self._widgets[name]
                    w.blockSignals(True)
                    w.setValue(item["default"])
                    w.blockSignals(False)
            elif wtype == "combo":
                if item["default"] in item["options"]:
                    w = self._widgets[name]
                    w.blockSignals(True)
                    w.setCurrentIndex(item["options"].index(item["default"]))
                    w.blockSignals(False)
