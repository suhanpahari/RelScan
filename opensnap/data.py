import glob
import os

import numpy as np
import torch

from . import paths

paths.add(paths.OPENSCENE_DIR)
try:
    from dataset.label_constants import SCANNET_LABELS_20, MATTERPORT_LABELS_21, MATTERPORT_LABELS_160, NUSCENES_LABELS_16  # noqa: E402
except ImportError:  # deployments without the OpenScene repo (e.g. the video demo)
    SCANNET_LABELS_20 = MATTERPORT_LABELS_21 = MATTERPORT_LABELS_160 = NUSCENES_LABELS_16 = ("other",)

LABELSETS = {
    "scannet": [*SCANNET_LABELS_20[:-1], "other"],
    "matterport": list(MATTERPORT_LABELS_21),
    "matterport160": list(MATTERPORT_LABELS_160),
    "nuscenes": list(NUSCENES_LABELS_16),
    "semantickitti": ["car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "bicyclist", "motorcyclist", "road",
                      "parking", "sidewalk", "other ground", "building", "fence", "vegetation", "trunk", "terrain", "pole",
                      "traffic sign"],
    "underonefacade": ["wall", "window", "door", "balcony", "molding", "decoration", "column", "arch", "stairs", "ground surface",
                       "terrain", "roof", "blinds", "interior", "other"],
}
DATA_DIRS = {"scannet": "scannet_3d", "matterport": "matterport_3d", "matterport160": "matterport_3d_160", "nuscenes": "nuscenes_official",
             "semantickitti": "semantickitti", "underonefacade": "underonefacade_5cm"}


def snap_train_vocab():
    """Class names SNAP's text head was supervised on (ScanNet200, SemanticKITTI, STPLS3D, DALES)."""
    f = os.path.join(os.path.dirname(__file__), "vocab", "snap_train.txt")
    if os.path.exists(f):
        return [l.strip() for l in open(f) if l.strip()]
    paths.add(paths.SNAP_DIR)
    from datasets.demo import DemoDatset
    d = DemoDatset.__new__(DemoDatset)
    DemoDatset.__init__(d, domain="Indoor")
    names = d.labels_scannet + d.labels_kitti + d.labels_stpls3d + d.labels_dales
    return sorted({n.lower().replace("_", " ") for n in names})


def scene_files(dataset, split, root=None):
    root = root or os.path.join(paths.DOWNLOADS, "data", DATA_DIRS[dataset])
    return sorted(glob.glob(os.path.join(root, split, "*.pth")))


def load_scene(path):
    coord, color, label = torch.load(path, weights_only=False)
    label = np.asarray(label).astype(np.int64)
    label[(label < 0) | (label == 255)] = 255
    color = None if np.isscalar(color) else (np.asarray(color) + 1.0) * 127.5
    return np.asarray(coord, np.float32), color, label


def estimate_normals(coord, knn=20):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coord.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
    pcd.orient_normals_towards_camera_location(coord.mean(0) + np.array([0, 0, 1.0]))
    return np.asarray(pcd.normals, np.float32)


def lvis_vocab():
    return [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "vocab", "lvis.txt")) if l.strip()]


def prior_vocab(name="snap"):
    """Name-proposal vocabulary: SNAP's training classes, optionally extended with generic LVIS names."""
    v = snap_train_vocab()
    return v if name == "snap" else sorted(set(v) | set(lvis_vocab()))
