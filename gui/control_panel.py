from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox, QPushButton,
    QSlider, QLabel, QStackedWidget, QCheckBox, QFormLayout, QHBoxLayout
)

from core.clustering import CLUSTERER_REGISTRY, CLUSTERER_NAMES_CN, CLUSTERER_KEYS
from core.data_generator import DataGenerator
from gui.params_widgets import BaseParamsWidget


class ControlPanel(QWidget):
    generate_clicked = pyqtSignal()
    run_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    algorithm_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(290)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Data generation group ---
        data_group = QGroupBox("数据生成")
        data_form = QFormLayout(data_group)

        self.shape_combo = QComboBox()
        for key in DataGenerator.ALL_SHAPES:
            self.shape_combo.addItem(DataGenerator.SHAPE_LABELS[key], key)
        data_form.addRow("数据形状:", self.shape_combo)

        self.samples_slider = QSlider(Qt.Horizontal)
        self.samples_slider.setRange(100, 2000)
        self.samples_slider.setValue(300)
        self.samples_slider.setSingleStep(50)
        self.samples_label = QLabel("300")
        self.samples_slider.valueChanged.connect(
            lambda v: self.samples_label.setText(str(v)))
        samples_row = QHBoxLayout()
        samples_row.addWidget(self.samples_slider)
        samples_row.addWidget(self.samples_label)
        data_form.addRow("样本数量:", samples_row)

        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(1, 50)
        self.noise_slider.setValue(10)
        self.noise_label = QLabel("0.10")
        self.noise_slider.valueChanged.connect(
            lambda v: self.noise_label.setText(f"{v / 100:.2f}"))
        noise_row = QHBoxLayout()
        noise_row.addWidget(self.noise_slider)
        noise_row.addWidget(self.noise_label)
        data_form.addRow("噪声水平:", noise_row)

        self.generate_btn = QPushButton("生成数据")
        self.generate_btn.clicked.connect(self.generate_clicked.emit)
        data_form.addRow(self.generate_btn)
        layout.addWidget(data_group)

        # --- Algorithm group ---
        algo_group = QGroupBox("聚类算法")
        algo_layout = QVBoxLayout(algo_group)

        algo_form = QFormLayout()
        self.algo_combo = QComboBox()
        for key in CLUSTERER_KEYS:
            self.algo_combo.addItem(CLUSTERER_NAMES_CN[key], key)
        algo_form.addRow("算法选择:", self.algo_combo)
        algo_layout.addLayout(algo_form)

        self.stacked = QStackedWidget()
        for key in CLUSTERER_KEYS:
            cls = CLUSTERER_REGISTRY[key]
            schema = cls().get_param_schema()
            widget = BaseParamsWidget(schema)
            self.stacked.addWidget(widget)
        algo_layout.addWidget(self.stacked)

        self.boundary_cb = QCheckBox("显示决策边界")
        algo_layout.addWidget(self.boundary_cb)

        self.run_btn = QPushButton("运行聚类")
        self.run_btn.clicked.connect(self.run_clicked.emit)
        algo_layout.addWidget(self.run_btn)

        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        layout.addWidget(algo_group)

        # --- Reset ---
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self.reset_btn)

        layout.addStretch()
        self._on_algo_changed(0)

    def _on_algo_changed(self, index):
        if index < 0 or index >= len(CLUSTERER_KEYS):
            return
        key = CLUSTERER_KEYS[index]
        self.stacked.setCurrentIndex(index)
        self.algorithm_changed.emit(key)

    def get_data_params(self):
        return {
            "shape": self.shape_combo.currentData(),
            "n_samples": self.samples_slider.value(),
            "noise": self.noise_slider.value() / 100.0,
        }

    def get_algorithm_key(self):
        return self.algo_combo.currentData()

    def get_algorithm_params(self):
        widget = self.stacked.currentWidget()
        return widget.get_params()

    def get_show_boundary(self):
        return self.boundary_cb.isChecked()

    def set_boundary_enabled(self, enabled):
        self.boundary_cb.setEnabled(enabled)
        if not enabled:
            self.boundary_cb.setChecked(False)

    def set_run_enabled(self, enabled):
        self.run_btn.setEnabled(enabled)

    def set_generate_enabled(self, enabled):
        self.generate_btn.setEnabled(enabled)

    def reset_defaults(self):
        self.shape_combo.setCurrentIndex(0)
        self.samples_slider.setValue(300)
        self.noise_slider.setValue(10)
        self.algo_combo.setCurrentIndex(0)
        for i in range(self.stacked.count()):
            w = self.stacked.widget(i)
            w.set_defaults()
        self.boundary_cb.setChecked(False)
        self.generate_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
