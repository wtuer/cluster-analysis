import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture


class BaseClusterer(ABC):
    name = ""
    name_cn = ""

    def __init__(self):
        self._params = self.get_default_params()
        self.labels_ = None
        self.execution_time_ = 0.0
        self._model = None

    @abstractmethod
    def get_default_params(self):
        ...

    @abstractmethod
    def get_param_schema(self):
        ...

    @abstractmethod
    def _build_and_fit(self, X):
        ...

    def set_params(self, **params):
        self._params.update(params)

    def fit(self, X):
        t0 = time.perf_counter()
        self.labels_ = self._build_and_fit(X)
        self.execution_time_ = (time.perf_counter() - t0) * 1000
        return self.labels_

    def get_centers(self):
        return None

    def has_centers(self):
        return False

    def has_predict(self):
        return False

    def get_model(self):
        return self._model if self.has_predict() else None


class KMeansClusterer(BaseClusterer):
    name = "K-Means"
    name_cn = "K-Means 聚类"

    def get_default_params(self):
        return {"n_clusters": 3, "n_init": 10}

    def get_param_schema(self):
        return [
            {"name": "n_clusters", "label_cn": "聚类数 K", "type": "int",
             "min": 2, "max": 10, "default": 3, "step": 1},
            {"name": "n_init", "label_cn": "初始化次数", "type": "int",
             "min": 1, "max": 20, "default": 10, "step": 1},
        ]

    def _build_and_fit(self, X):
        self._model = KMeans(**self._params, random_state=42)
        return self._model.fit_predict(X)

    def get_centers(self):
        if self._model is not None:
            return self._model.cluster_centers_
        return None

    def has_centers(self):
        return True

    def has_predict(self):
        return True


class DBSCANClusterer(BaseClusterer):
    name = "DBSCAN"
    name_cn = "DBSCAN 密度聚类"

    def get_default_params(self):
        return {"eps": 0.5, "min_samples": 5}

    def get_param_schema(self):
        return [
            {"name": "eps", "label_cn": "邻域半径 ε", "type": "float",
             "min": 0.05, "max": 3.0, "default": 0.5, "step": 0.05},
            {"name": "min_samples", "label_cn": "最小样本数", "type": "int",
             "min": 2, "max": 20, "default": 5, "step": 1},
        ]

    def _build_and_fit(self, X):
        self._model = DBSCAN(**self._params)
        return self._model.fit_predict(X)


class HierarchicalClusterer(BaseClusterer):
    name = "Hierarchical"
    name_cn = "层次聚类 (AGNES)"

    def get_default_params(self):
        return {"n_clusters": 3, "linkage": "ward"}

    def get_param_schema(self):
        return [
            {"name": "n_clusters", "label_cn": "聚类数", "type": "int",
             "min": 2, "max": 10, "default": 3, "step": 1},
            {"name": "linkage", "label_cn": "链接方式", "type": "combo",
             "options": ["ward", "complete", "average", "single"],
             "option_labels_cn": ["Ward", "完全链接", "平均链接", "单链接"],
             "default": "ward"},
        ]

    def _build_and_fit(self, X):
        self._model = AgglomerativeClustering(**self._params)
        return self._model.fit_predict(X)


class MeanShiftClusterer(BaseClusterer):
    name = "MeanShift"
    name_cn = "均值漂移聚类"

    def get_default_params(self):
        return {"bandwidth": None}

    def get_param_schema(self):
        return [
            {"name": "bandwidth", "label_cn": "带宽", "type": "float",
             "min": 0.1, "max": 10.0, "default": None, "step": 0.1,
             "nullable": True, "nullable_label": "自动估计带宽"},
        ]

    def _build_and_fit(self, X):
        bw = self._params.get("bandwidth")
        if bw is None:
            bw = estimate_bandwidth(X, quantile=0.2)
            if bw <= 0:
                bw = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = MeanShift(bandwidth=bw, bin_seeding=True)
            return self._model.fit_predict(X)

    def get_centers(self):
        if self._model is not None:
            return self._model.cluster_centers_
        return None

    def has_centers(self):
        return True

    def has_predict(self):
        return True


class GMMClusterer(BaseClusterer):
    name = "GMM"
    name_cn = "高斯混合模型 (GMM)"

    def get_default_params(self):
        return {"n_components": 3, "covariance_type": "full"}

    def get_param_schema(self):
        return [
            {"name": "n_components", "label_cn": "成分数", "type": "int",
             "min": 2, "max": 10, "default": 3, "step": 1},
            {"name": "covariance_type", "label_cn": "协方差类型", "type": "combo",
             "options": ["full", "tied", "diag", "spherical"],
             "option_labels_cn": ["完全协方差", "绑定协方差", "对角协方差", "球形协方差"],
             "default": "full"},
        ]

    def _build_and_fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = GaussianMixture(**self._params, random_state=42)
            self._model.fit(X)
            return self._model.predict(X)

    def get_centers(self):
        if self._model is not None:
            return self._model.means_
        return None

    def has_centers(self):
        return True

    def has_predict(self):
        return True


class SpectralClusterer(BaseClusterer):
    name = "Spectral"
    name_cn = "谱聚类"

    def get_default_params(self):
        return {"n_clusters": 3, "affinity": "rbf"}

    def get_param_schema(self):
        return [
            {"name": "n_clusters", "label_cn": "聚类数", "type": "int",
             "min": 2, "max": 10, "default": 3, "step": 1},
            {"name": "affinity", "label_cn": "亲和度", "type": "combo",
             "options": ["rbf", "nearest_neighbors"],
             "option_labels_cn": ["RBF 核", "最近邻"],
             "default": "rbf"},
        ]

    def _build_and_fit(self, X):
        params = dict(self._params)
        if params.get("affinity") == "nearest_neighbors":
            params["n_neighbors"] = 10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SpectralClustering(**params, random_state=42)
            return self._model.fit_predict(X)


CLUSTERER_REGISTRY = {
    "kmeans": KMeansClusterer,
    "dbscan": DBSCANClusterer,
    "hierarchical": HierarchicalClusterer,
    "meanshift": MeanShiftClusterer,
    "gmm": GMMClusterer,
    "spectral": SpectralClusterer,
}

CLUSTERER_NAMES_CN = {
    "kmeans": "K-Means 聚类",
    "dbscan": "DBSCAN 密度聚类",
    "hierarchical": "层次聚类 (AGNES)",
    "meanshift": "均值漂移聚类",
    "gmm": "高斯混合模型 (GMM)",
    "spectral": "谱聚类",
}

CLUSTERER_KEYS = list(CLUSTERER_REGISTRY.keys())
