import numpy as np
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
NOISE_COLOR = "#cccccc"


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, figsize=(8, 6)):
        self.figure = Figure(figsize=figsize, dpi=100)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.tight_layout(pad=2.0)

    def clear(self):
        self.axes.clear()
        self.axes.set_xlabel("")
        self.axes.set_ylabel("")
        self.axes.set_title("")
        self.draw()

    def plot_data(self, X, y_true=None, title="原始数据"):
        self.axes.clear()
        if y_true is not None:
            unique_labels = np.unique(y_true)
            for i, label in enumerate(unique_labels):
                mask = y_true == label
                self.axes.scatter(
                    X[mask, 0], X[mask, 1],
                    c=COLORS[i % len(COLORS)], s=20, alpha=0.7,
                    label=f"类别 {label}", edgecolors="none"
                )
            self.axes.legend(loc="best", fontsize=8)
        else:
            self.axes.scatter(X[:, 0], X[:, 1], c="#1f77b4", s=20, alpha=0.7, edgecolors="none")
        self.axes.set_title(title, fontsize=12)
        self.axes.set_xlabel("X₁")
        self.axes.set_ylabel("X₂")
        self.figure.tight_layout(pad=2.0)
        self.draw()

    def plot_clusters(self, X, labels, centers=None, show_boundary=False, model=None, title="聚类结果"):
        self.axes.clear()
        unique_labels = sorted(set(labels))
        has_noise = -1 in unique_labels

        if has_noise:
            noise_mask = labels == -1
            self.axes.scatter(
                X[noise_mask, 0], X[noise_mask, 1],
                c=NOISE_COLOR, s=15, marker="x", alpha=0.5, label="噪声点"
            )
            unique_labels.remove(-1)

        if show_boundary and model is not None and hasattr(model, "predict"):
            self._plot_decision_boundary(X, model, unique_labels)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            self.axes.scatter(
                X[mask, 0], X[mask, 1],
                c=COLORS[i % len(COLORS)], s=25, alpha=0.7,
                label=f"簇 {label}", edgecolors="none"
            )

        if centers is not None and len(centers) > 0:
            self.axes.scatter(
                centers[:, 0], centers[:, 1],
                c="red", marker="D", s=80, edgecolors="black",
                linewidths=1.0, zorder=10, label="聚类中心"
            )

        self.axes.set_title(title, fontsize=12)
        self.axes.set_xlabel("X₁")
        self.axes.set_ylabel("X₂")
        self.axes.legend(loc="best", fontsize=8)
        self.figure.tight_layout(pad=2.0)
        self.draw()

    def _plot_decision_boundary(self, X, model, unique_labels):
        try:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            x_range = x_max - x_min
            y_range = y_max - y_min
            n_steps = 200
            h_x = max(x_range / n_steps, 0.01)
            h_y = max(y_range / n_steps, 0.01)
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, h_x),
                np.arange(y_min, y_max, h_y)
            )
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            n_colors = max(len(unique_labels), int(Z.max()) + 1)
            cmap = ListedColormap(COLORS[:n_colors])
            self.axes.contourf(xx, yy, Z, alpha=0.12, cmap=cmap,
                               levels=np.arange(-0.5, n_colors + 0.5, 1))
        except Exception:
            pass
