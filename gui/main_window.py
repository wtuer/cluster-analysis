import traceback

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStatusBar, QSplitter,
    QMessageBox
)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from core.clustering import CLUSTERER_REGISTRY, CLUSTERER_KEYS
from core.data_generator import DataGenerator
from gui.control_panel import ControlPanel
from gui.plot_canvas import PlotCanvas
from utils.metrics import compute_all_metrics


class ClusteringWorker(QThread):
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, clusterer, X, parent=None):
        super().__init__(parent)
        self.clusterer = clusterer
        self.X = X

    def run(self):
        try:
            labels = self.clusterer.fit(self.X)
            metrics = compute_all_metrics(self.X, labels)
            metrics["execution_time"] = self.clusterer.execution_time_
            self.finished.emit(labels, metrics)
        except Exception:
            self.error.emit(traceback.format_exc())


PREDICTABLE_KEYS = {"kmeans", "gmm", "meanshift"}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.X = None
        self.y_true = None
        self._clusterers = {}
        self._current_clusterer = None
        self._worker = None

        self._init_clusterers()
        self._init_ui()
        self._connect_signals()

    def _init_clusterers(self):
        for key in CLUSTERER_KEYS:
            self._clusterers[key] = CLUSTERER_REGISTRY[key]()

    def _init_ui(self):
        self.setWindowTitle("聚类算法可视化工具")
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 请先生成数据")

        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        h_layout.addWidget(splitter)

        self.control_panel = ControlPanel()
        splitter.addWidget(self.control_panel)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = PlotCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _connect_signals(self):
        self.control_panel.generate_clicked.connect(self._on_generate)
        self.control_panel.run_clicked.connect(self._on_run)
        self.control_panel.reset_clicked.connect(self._on_reset)
        self.control_panel.algorithm_changed.connect(self._on_algorithm_changed)

    def _on_generate(self):
        try:
            params = self.control_panel.get_data_params()
            gen = DataGenerator()
            self.X, self.y_true = gen.generate(
                shape=params["shape"],
                n_samples=params["n_samples"],
                noise=params["noise"],
                random_state=42,
            )
            self.canvas.plot_data(self.X, self.y_true)
            shape_label = DataGenerator.SHAPE_LABELS[params["shape"]]
            self.status_bar.showMessage(
                f"已生成 {len(self.X)} 个样本 — {shape_label}"
            )
        except Exception as e:
            QMessageBox.warning(self, "生成失败", f"数据生成出错:\n{e}")

    def _stop_worker(self):
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
                self._worker.terminate()
                self._worker.wait()
            self._worker = None

    def _on_run(self):
        if self.X is None:
            QMessageBox.warning(self, "提示", "请先生成数据！")
            return

        self._stop_worker()

        key = self.control_panel.get_algorithm_key()
        params = self.control_panel.get_algorithm_params()

        clusterer = self._clusterers[key]
        clusterer.set_params(**params)
        self._current_clusterer = clusterer
        self._run_X = self.X.copy()

        self.control_panel.set_run_enabled(False)
        self.control_panel.set_generate_enabled(False)
        self.status_bar.showMessage("正在计算...")

        self._worker = ClusteringWorker(clusterer, self._run_X, self)
        self._worker.finished.connect(self._on_clustering_done)
        self._worker.error.connect(self._on_clustering_error)
        self._worker.start()

    def _on_clustering_done(self, labels, metrics):
        self.control_panel.set_run_enabled(True)
        self.control_panel.set_generate_enabled(True)

        if self._current_clusterer is None:
            return

        X = self._run_X
        centers = self._current_clusterer.get_centers() if self._current_clusterer.has_centers() else None
        show_boundary = self.control_panel.get_show_boundary()
        model = self._current_clusterer.get_model() if show_boundary else None

        self.canvas.plot_clusters(
            X, labels, centers=centers,
            show_boundary=show_boundary, model=model
        )

        parts = [f"聚类数: {metrics['n_clusters']}"]
        if metrics["silhouette"] >= 0:
            parts.append(f"轮廓系数: {metrics['silhouette']:.3f}")
        parts.append(f"耗时: {metrics['execution_time']:.2f}ms")
        if metrics["n_noise"] > 0:
            parts.append(f"噪声点: {metrics['n_noise']}")
        self.status_bar.showMessage(" | ".join(parts))

    def _on_clustering_error(self, error_msg):
        self.control_panel.set_run_enabled(True)
        self.control_panel.set_generate_enabled(True)
        self.status_bar.showMessage("聚类计算失败")
        QMessageBox.critical(self, "计算错误", f"聚类算法执行出错:\n\n{error_msg[:500]}")

    def _on_algorithm_changed(self, key):
        self.control_panel.set_boundary_enabled(key in PREDICTABLE_KEYS)

    def _on_reset(self):
        self._stop_worker()
        self.X = None
        self.y_true = None
        self._current_clusterer = None
        self.canvas.clear()
        self.control_panel.reset_defaults()
        self.status_bar.showMessage("已重置 — 请生成数据")

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()
