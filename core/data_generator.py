import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles


class DataGenerator:
    SHAPE_BLOBS = "blobs"
    SHAPE_MOONS = "moons"
    SHAPE_CIRCLES = "circles"
    SHAPE_ANISO = "aniso"
    SHAPE_VARIED = "varied"

    ALL_SHAPES = [SHAPE_BLOBS, SHAPE_MOONS, SHAPE_CIRCLES, SHAPE_ANISO, SHAPE_VARIED]

    SHAPE_LABELS = {
        SHAPE_BLOBS: "高斯聚类 (Blobs)",
        SHAPE_MOONS: "双月形 (Moons)",
        SHAPE_CIRCLES: "同心圆 (Circles)",
        SHAPE_ANISO: "各向异性 (Anisotropic)",
        SHAPE_VARIED: "不同密度 (Varied)",
    }

    def generate(self, shape, n_samples=300, noise=0.1, random_state=None):
        generators = {
            self.SHAPE_BLOBS: self._make_blobs,
            self.SHAPE_MOONS: self._make_moons,
            self.SHAPE_CIRCLES: self._make_circles,
            self.SHAPE_ANISO: self._make_aniso,
            self.SHAPE_VARIED: self._make_varied,
        }
        if shape not in generators:
            raise ValueError(f"Unknown shape: {shape}")
        return generators[shape](n_samples, noise, random_state)

    def _make_blobs(self, n_samples, noise, random_state):
        cluster_std = 0.5 + noise * 2.0
        X, y = make_blobs(
            n_samples=n_samples, centers=3,
            cluster_std=cluster_std, random_state=random_state
        )
        return X, y

    def _make_moons(self, n_samples, noise, random_state):
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        return X, y

    def _make_circles(self, n_samples, noise, random_state):
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state
        )
        return X, y

    def _make_aniso(self, n_samples, noise, random_state):
        cluster_std = 0.5 + noise * 2.0
        X, y = make_blobs(
            n_samples=n_samples, centers=3,
            cluster_std=cluster_std, random_state=random_state
        )
        transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
        X = np.dot(X, transformation)
        return X, y

    def _make_varied(self, n_samples, noise, random_state):
        cluster_std = [1.0 + noise, 2.5 + noise * 2, 0.5 + noise * 0.5]
        X, y = make_blobs(
            n_samples=n_samples, centers=3,
            cluster_std=cluster_std, random_state=random_state
        )
        return X, y
