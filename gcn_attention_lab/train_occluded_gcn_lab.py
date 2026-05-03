import argparse
import json
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mediapipe.tasks.python.vision import FaceLandmarksConnections
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm import tqdm

NODE_FEAT_DIM = 38
GEOMETRY_START_INDEX = 2
GEOMETRY_END_INDEX = 13
APPEARANCE_START_INDEX = 13
APPEARANCE_END_INDEX = 35
RELIABILITY_INDEX = 35
VALID_FLAG_INDEX = 36
OCC_FLAG_INDEX = 37
TOPOLOGY_EDGE_WEIGHT = 1.0
KNN_EDGE_WEIGHT = 0.35
OCCLUDED_MESSAGE_SCALE = 0.2
FACE_REGION_CONNECTIONS = {
    "face_oval": "FACE_LANDMARKS_FACE_OVAL",
    "left_eye": "FACE_LANDMARKS_LEFT_EYE",
    "right_eye": "FACE_LANDMARKS_RIGHT_EYE",
    "left_iris": "FACE_LANDMARKS_LEFT_IRIS",
    "right_iris": "FACE_LANDMARKS_RIGHT_IRIS",
    "left_eyebrow": "FACE_LANDMARKS_LEFT_EYEBROW",
    "right_eyebrow": "FACE_LANDMARKS_RIGHT_EYEBROW",
    "lips": "FACE_LANDMARKS_LIPS",
    "nose": "FACE_LANDMARKS_NOSE",
}
REGION_NAMES = tuple(FACE_REGION_CONNECTIONS.keys())
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MXREC_PREFIX = "mxrec://"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SampleMeta:
    image_path: str
    label: int
    class_name: str


class MXIndexedRecordReader:
    MAGIC = 0xCED7230A
    HEADER_SIZE = 28
    PREFIX_SIZE = 4

    def __init__(self, root: Path):
        self.root = root
        self.rec_path = root / "train.rec"
        self.idx_path = root / "train.idx"
        if not self.rec_path.exists() or not self.idx_path.exists():
            raise FileNotFoundError(f"MXNet RecordIO files not found under: {root}")

        self.offsets: Dict[int, int] = {}
        with open(self.idx_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.offsets[int(parts[0])] = int(parts[1])
        self.sorted_keys = sorted(self.offsets)
        self.next_offsets: Dict[int, int] = {}
        rec_size = self.rec_path.stat().st_size
        for i, key in enumerate(self.sorted_keys):
            self.next_offsets[key] = self.offsets[self.sorted_keys[i + 1]] if i + 1 < len(self.sorted_keys) else rec_size
        self._fh = None

    def _file(self):
        if self._fh is None:
            self._fh = open(self.rec_path, "rb")
        return self._fh

    def read_image(self, record_key: int) -> Optional[np.ndarray]:
        offset = self.offsets.get(record_key)
        if offset is None:
            return None

        fh = self._file()
        fh.seek(offset)
        magic_raw = fh.read(self.PREFIX_SIZE)
        if len(magic_raw) != self.PREFIX_SIZE:
            return None
        magic = struct.unpack("<I", magic_raw)[0]
        if magic != self.MAGIC:
            return None

        header = fh.read(self.HEADER_SIZE)
        if len(header) != self.HEADER_SIZE:
            return None

        data_len = self.next_offsets[record_key] - offset - self.PREFIX_SIZE - self.HEADER_SIZE
        if data_len <= 0:
            return None
        data = fh.read(data_len)
        img_arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def scan_lfw(root: Path, min_images_per_identity: int, max_classes: int, max_images_per_class: int) -> Tuple[List[SampleMeta], Dict[str, int]]:
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda x: x.name)

    selected = []
    for d in class_dirs:
        imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if len(imgs) >= min_images_per_identity:
            selected.append((d.name, imgs))

    selected.sort(key=lambda x: len(x[1]), reverse=True)
    if max_classes > 0:
        selected = selected[:max_classes]

    class_to_idx = {name: i for i, (name, _) in enumerate(selected)}
    samples: List[SampleMeta] = []

    for class_name, imgs in selected:
        if max_images_per_class > 0:
            imgs = imgs[:max_images_per_class]
        for p in imgs:
            samples.append(SampleMeta(str(p), class_to_idx[class_name], class_name))

    return samples, class_to_idx


def scan_mxrec(root: Path, min_images_per_identity: int, max_classes: int, max_images_per_class: int) -> Tuple[List[SampleMeta], Dict[str, int]]:
    lst_path = root / "train.lst"
    if not lst_path.exists():
        raise FileNotFoundError(f"MXNet RecordIO list file not found: {lst_path}")

    by_class: Dict[str, List[int]] = {}
    with open(lst_path, "r", encoding="utf-8") as f:
        for record_key, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            class_name = parts[2] if len(parts) >= 3 else Path(parts[1].replace("\\", "/")).parent.name
            by_class.setdefault(class_name, []).append(record_key)

    selected = [(name, keys) for name, keys in by_class.items() if len(keys) >= min_images_per_identity]
    selected.sort(key=lambda x: len(x[1]), reverse=True)
    if max_classes > 0:
        selected = selected[:max_classes]

    class_to_idx = {name: i for i, (name, _) in enumerate(selected)}
    samples: List[SampleMeta] = []
    for class_name, keys in selected:
        if max_images_per_class > 0:
            keys = keys[:max_images_per_class]
        label = class_to_idx[class_name]
        for key in keys:
            samples.append(SampleMeta(f"{MXREC_PREFIX}{key}", label, class_name))

    return samples, class_to_idx


def split_by_identity(samples: Sequence[SampleMeta], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[SampleMeta], List[SampleMeta], List[SampleMeta]]:
    by_class: Dict[int, List[SampleMeta]] = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for _, arr in by_class.items():
        rng.shuffle(arr)
        n = len(arr)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_val = 1
            n_train = max(1, n - 2)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train.extend(arr[:n_train])
        val.extend(arr[n_train:n_train + n_val])
        test.extend(arr[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class LandmarkExtractor:
    def __init__(self, num_nodes: int = 468):
        self.num_nodes = num_nodes
        self.semantic_groups = self._build_semantic_groups(num_nodes)
        self.landmarker = None

        model_path = PROJECT_ROOT / "models" / "face_landmarker.task"
        if model_path.exists() and hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        elif num_nodes == 468:
            raise RuntimeError("468-node face graph requires MediaPipe Face Landmarker and models/face_landmarker.task.")

    @staticmethod
    def _build_semantic_groups(num_nodes: int) -> List[List[int]]:
        if num_nodes == 468:
            return [[idx] for idx in range(468)]

        raise ValueError("Current implementation is aligned to the 468-node MediaPipe face graph.")

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.landmarker is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                coords = np.zeros((len(self.semantic_groups), 2), dtype=np.float32)
                h, w = image_bgr.shape[:2]

                for i, group in enumerate(self.semantic_groups):
                    xs = [np.clip(lm[idx].x, 0.0, 1.0) for idx in group]
                    ys = [np.clip(lm[idx].y, 0.0, 1.0) for idx in group]
                    coords[i, 0] = float(np.mean(xs)) * (w - 1)
                    coords[i, 1] = float(np.mean(ys)) * (h - 1)
                return coords

        return np.full((self.num_nodes, 2), -1.0, dtype=np.float32)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    deg = torch.sum(adj, dim=-1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    d = torch.diag_embed(deg_inv_sqrt)
    return d @ adj @ d


def _connections_to_edges(connection_name: str, num_nodes: int) -> List[Tuple[int, int]]:
    connections = getattr(FaceLandmarksConnections, connection_name)
    edges = []
    for conn in connections:
        u, v = int(conn.start), int(conn.end)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            edges.append((u, v))
    return edges


def build_topology_adjacency(num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in _connections_to_edges("FACE_LANDMARKS_TESSELATION", num_nodes):
        adj[u, v] = TOPOLOGY_EDGE_WEIGHT
        adj[v, u] = TOPOLOGY_EDGE_WEIGHT
    np.fill_diagonal(adj, 1.0)
    return adj


def build_face_graph_adjacency(coords_xy: np.ndarray, k: int) -> np.ndarray:
    n = coords_xy.shape[0]
    adj = build_topology_adjacency(n)
    valid = np.where(coords_xy[:, 0] >= 0)[0]

    if len(valid) <= 1 or k <= 0:
        return adj

    pts = coords_xy[valid]
    dist = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    for i in range(len(valid)):
        nn_idx = np.argsort(dist[i])[1 : k + 1]
        for j in nn_idx:
            u = valid[i]
            v = valid[j]
            adj[u, v] = max(adj[u, v], KNN_EDGE_WEIGHT)
            adj[v, u] = max(adj[v, u], KNN_EDGE_WEIGHT)

    np.fill_diagonal(adj, 1.0)
    return adj


def build_graph_adjacency(coords_xy: np.ndarray, k: int, graph_mode: str) -> np.ndarray:
    n = coords_xy.shape[0]
    if graph_mode == "self":
        return np.eye(n, dtype=np.float32)
    if graph_mode == "topology":
        return build_topology_adjacency(n)
    if graph_mode == "knn":
        adj = np.zeros((n, n), dtype=np.float32)
        valid = np.where(coords_xy[:, 0] >= 0)[0]
        if len(valid) > 1 and k > 0:
            pts = coords_xy[valid]
            dist = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
            for i in range(len(valid)):
                nn_idx = np.argsort(dist[i])[1 : k + 1]
                for j in nn_idx:
                    u = valid[i]
                    v = valid[j]
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
        np.fill_diagonal(adj, 1.0)
        return adj
    if graph_mode == "topology_knn":
        return build_face_graph_adjacency(coords_xy, k)
    raise ValueError(f"Unsupported graph_mode: {graph_mode}")


def build_region_masks(num_nodes: int) -> torch.Tensor:
    masks = []
    for connection_name in FACE_REGION_CONNECTIONS.values():
        nodes = sorted({idx for edge in _connections_to_edges(connection_name, num_nodes) for idx in edge})
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        if nodes:
            mask[nodes] = True
        masks.append(mask)

    full_face = torch.ones(num_nodes, dtype=torch.bool)
    masks.append(full_face)
    return torch.stack(masks, dim=0)


def build_semantic_region_masks(num_nodes: int) -> torch.Tensor:
    masks = []
    for connection_name in FACE_REGION_CONNECTIONS.values():
        nodes = sorted({idx for edge in _connections_to_edges(connection_name, num_nodes) for idx in edge})
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        if nodes:
            mask[nodes] = True
        masks.append(mask)
    return torch.stack(masks, dim=0)


def build_region_adjacency() -> torch.Tensor:
    region_count = len(REGION_NAMES)
    adj = torch.zeros(region_count, region_count, dtype=torch.float32)
    region_to_idx = {name: idx for idx, name in enumerate(REGION_NAMES)}
    edges = [
        ("face_oval", "left_eye"),
        ("face_oval", "right_eye"),
        ("face_oval", "left_eyebrow"),
        ("face_oval", "right_eyebrow"),
        ("face_oval", "nose"),
        ("left_eye", "left_eyebrow"),
        ("right_eye", "right_eyebrow"),
        ("left_eye", "left_iris"),
        ("right_eye", "right_iris"),
        ("left_eye", "nose"),
        ("right_eye", "nose"),
        ("left_eyebrow", "nose"),
        ("right_eyebrow", "nose"),
        ("nose", "lips"),
        ("left_eye", "right_eye"),
        ("left_iris", "right_iris"),
        ("left_eyebrow", "right_eyebrow"),
    ]
    for left, right in edges:
        u = region_to_idx[left]
        v = region_to_idx[right]
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    adj.fill_diagonal_(1.0)
    return adj


def build_topology_neighbors(num_nodes: int) -> List[List[int]]:
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in _connections_to_edges("FACE_LANDMARKS_TESSELATION", num_nodes):
        neighbors[u].add(v)
        neighbors[v].add(u)
    return [sorted(arr) for arr in neighbors]


def normalize_masked_adjacency(adj: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
    valid = node_feat[:, :, VALID_FLAG_INDEX:VALID_FLAG_INDEX + 1]
    reliability = node_feat[:, :, RELIABILITY_INDEX:RELIABILITY_INDEX + 1]
    occ = node_feat[:, :, OCC_FLAG_INDEX:OCC_FLAG_INDEX + 1]
    source_reliability = (
        valid.transpose(1, 2)
        * reliability.transpose(1, 2)
        * (1.0 - occ.transpose(1, 2) * (1.0 - OCCLUDED_MESSAGE_SCALE))
    )
    target_valid = valid
    masked_adj = adj * source_reliability * target_valid

    eye = torch.eye(adj.size(-1), dtype=adj.dtype, device=adj.device).unsqueeze(0)
    masked_adj = torch.maximum(masked_adj, eye)
    return normalize_adjacency(masked_adj)


def patch_entropy(patch_gray: np.ndarray, bins: int = 8) -> float:
    hist, _ = np.histogram(patch_gray, bins=bins, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float32) / max(float(np.sum(hist)), 1.0)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)) / np.log2(bins))


def local_binary_pattern_stats(image_gray: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    h, w = image_gray.shape[:2]
    center = image_gray[y, x]
    offsets = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    bits = []
    for dx, dy in offsets:
        xx = int(np.clip(x + dx, 0, w - 1))
        yy = int(np.clip(y + dy, 0, h - 1))
        bits.append(1.0 if image_gray[yy, xx] >= center else 0.0)

    transitions = sum(1 for i in range(len(bits)) if bits[i] != bits[(i + 1) % len(bits)])
    return float(np.mean(bits)), float(transitions / len(bits))


def normalize_feature_array(values: np.ndarray, valid: np.ndarray, invert: bool = False) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    valid_values = values[valid]
    if valid_values.size == 0:
        return out

    lo = float(np.percentile(valid_values, 10))
    hi = float(np.percentile(valid_values, 90))
    if hi - lo < 1e-6:
        out[valid] = 1.0
    else:
        out[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        out[valid] = 1.0 - out[valid]
    return out


def make_node_features(image_bgr: np.ndarray, coords_xy: np.ndarray, rgb_window_size: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    n = coords_xy.shape[0]
    feat = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    valid = (coords_xy[:, 0] >= 0) & (coords_xy[:, 1] >= 0)
    if not np.any(valid):
        return feat

    x_norm = coords_xy[:, 0] / max(w - 1, 1)
    y_norm = coords_xy[:, 1] / max(h - 1, 1)
    center_x = float(np.mean(x_norm[valid]))
    center_y = float(np.mean(y_norm[valid]))
    min_x, max_x = float(np.min(x_norm[valid])), float(np.max(x_norm[valid]))
    min_y, max_y = float(np.min(y_norm[valid])), float(np.max(y_norm[valid]))
    bbox_w = max(max_x - min_x, 1e-6)
    bbox_h = max(max_y - min_y, 1e-6)
    face_scale = max(float(np.sqrt(bbox_w * bbox_h)), 1e-6)
    rel_x = x_norm - center_x
    rel_y = y_norm - center_y
    radius = np.sqrt(rel_x ** 2 + rel_y ** 2)
    angle = np.arctan2(rel_y, rel_x)
    half_window = max(1, rgb_window_size // 2)
    global_rgb_mean = np.mean(image_rgb, axis=(0, 1))
    global_gray_mean = float(np.mean(image_gray))
    mean_grad_arr = np.zeros(n, dtype=np.float32)
    std_gray_arr = np.zeros(n, dtype=np.float32)
    entropy_arr = np.zeros(n, dtype=np.float32)
    border_arr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if not valid[i]:
            feat[i] = 0.0
            continue

        xi = int(np.clip(round(coords_xy[i, 0]), 0, w - 1))
        yi = int(np.clip(round(coords_xy[i, 1]), 0, h - 1))
        x0 = max(0, xi - half_window)
        x1 = min(w, xi + half_window + 1)
        y0 = max(0, yi - half_window)
        y1 = min(h, yi + half_window + 1)
        patch_rgb = image_rgb[y0:y1, x0:x1]
        patch_gray = image_gray[y0:y1, x0:x1]
        patch_grad = grad_mag[y0:y1, x0:x1]
        center_gray = image_gray[yi, xi]
        lbp_density, lbp_transition = local_binary_pattern_stats(image_gray, xi, yi)

        feat[i, 0] = x_norm[i]
        feat[i, 1] = y_norm[i]
        feat[i, 2] = rel_x[i]
        feat[i, 3] = rel_y[i]
        feat[i, 4] = radius[i]
        feat[i, 5] = np.sin(angle[i])
        feat[i, 6] = np.cos(angle[i])
        feat[i, 7] = (x_norm[i] - min_x) / bbox_w
        feat[i, 8] = (y_norm[i] - min_y) / bbox_h
        feat[i, 9] = rel_x[i] / face_scale
        feat[i, 10] = rel_y[i] / face_scale
        feat[i, 11] = np.sin(2.0 * np.pi * i / max(n, 1))
        feat[i, 12] = np.cos(2.0 * np.pi * i / max(n, 1))

        mean_rgb = np.mean(patch_rgb, axis=(0, 1))
        std_rgb = np.std(patch_rgb, axis=(0, 1))
        mean_gray = float(np.mean(patch_gray))
        std_gray = float(np.std(patch_gray))
        mean_grad = float(np.mean(patch_grad))
        entropy = patch_entropy(patch_gray)
        border_margin = min(x_norm[i], 1.0 - x_norm[i], y_norm[i], 1.0 - y_norm[i])
        feat[i, 13:16] = mean_rgb
        feat[i, 16:19] = std_rgb
        feat[i, 19] = mean_gray
        feat[i, 20] = std_gray
        feat[i, 21] = mean_grad
        feat[i, 22] = float(np.std(patch_grad))
        feat[i, 23] = float(np.mean(np.abs(patch_gray - center_gray)))
        feat[i, 24] = center_gray
        feat[i, 25:28] = mean_rgb - global_rgb_mean
        feat[i, 28] = mean_gray - global_gray_mean
        feat[i, 29] = center_gray - global_gray_mean
        feat[i, 30] = lbp_density
        feat[i, 31] = lbp_transition
        feat[i, 32] = float(np.mean(np.abs(grad_x[y0:y1, x0:x1])))
        feat[i, 33] = float(np.mean(np.abs(grad_y[y0:y1, x0:x1])))
        feat[i, 34] = entropy
        mean_grad_arr[i] = mean_grad
        std_gray_arr[i] = std_gray
        entropy_arr[i] = entropy
        border_arr[i] = np.clip(border_margin / 0.15, 0.0, 1.0)
        feat[i, OCC_FLAG_INDEX] = 0.0

    density_residual = np.full(n, 1e6, dtype=np.float32)
    topology_residual = np.full(n, 1e6, dtype=np.float32)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 1:
        valid_coords = np.stack([x_norm[valid], y_norm[valid]], axis=1)
        dist = np.sqrt(np.sum((valid_coords[:, None, :] - valid_coords[None, :, :]) ** 2, axis=2))
        for i, node_idx in enumerate(valid_idx):
            nn = np.sort(dist[i])[1 : min(4, len(valid_idx))]
            if nn.size > 0:
                density_residual[node_idx] = float(np.mean(nn) / face_scale)

    topology_neighbors = build_topology_neighbors(n)
    for i in valid_idx:
        neigh = [j for j in topology_neighbors[i] if valid[j]]
        if not neigh:
            continue
        neigh_coords = np.stack([x_norm[neigh], y_norm[neigh]], axis=1)
        center = np.array([x_norm[i], y_norm[i]], dtype=np.float32)
        topology_residual[i] = float(np.mean(np.sqrt(np.sum((neigh_coords - center) ** 2, axis=1))) / face_scale)

    grad_score = normalize_feature_array(mean_grad_arr, valid)
    texture_score = 0.5 * normalize_feature_array(std_gray_arr, valid) + 0.5 * normalize_feature_array(entropy_arr, valid)
    density_score = normalize_feature_array(density_residual, valid, invert=True)
    topology_score = normalize_feature_array(topology_residual, valid, invert=True)
    reliability = np.zeros(n, dtype=np.float32)
    reliability[valid] = (
        0.35 * topology_score[valid]
        + 0.25 * density_score[valid]
        + 0.20 * grad_score[valid]
        + 0.10 * texture_score[valid]
        + 0.10 * border_arr[valid]
    )
    reliability[valid] = np.clip(reliability[valid], 0.05, 1.0)
    feat[:, RELIABILITY_INDEX] = reliability
    feat[valid, VALID_FLAG_INDEX] = 1.0

    return feat


def sample_occlusion_mask(node_feat: np.ndarray) -> np.ndarray:
    x = node_feat[:, 0]
    y = node_feat[:, 1]
    valid = (node_feat[:, VALID_FLAG_INDEX] > 0.5) & (node_feat[:, OCC_FLAG_INDEX] < 0.5)
    mask = np.zeros(node_feat.shape[0], dtype=bool)

    if not np.any(valid):
        return mask

    mode = random.choice(["random_block", "lower_face", "eye_band", "left_side", "right_side"])

    if mode == "random_block":
        w = random.uniform(0.20, 0.40)
        h = random.uniform(0.20, 0.40)
        x0 = random.uniform(0.0, 1.0 - w)
        y0 = random.uniform(0.0, 1.0 - h)
        x1, y1 = x0 + w, y0 + h
        mask = valid & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    elif mode == "lower_face":
        y0 = random.uniform(0.52, 0.68)
        mask = valid & (y >= y0)
    elif mode == "eye_band":
        yc = random.uniform(0.30, 0.42)
        half_h = random.uniform(0.06, 0.12)
        mask = valid & (y >= yc - half_h) & (y <= yc + half_h)
    elif mode == "left_side":
        x1 = random.uniform(0.30, 0.45)
        mask = valid & (x <= x1)
    elif mode == "right_side":
        x0 = random.uniform(0.55, 0.70)
        mask = valid & (x >= x0)

    return mask


def apply_random_occlusion(node_feat: np.ndarray, occlusion_prob: float) -> np.ndarray:
    out = node_feat.copy()
    if random.random() > occlusion_prob:
        return out

    inside = sample_occlusion_mask(out)
    if not np.any(inside):
        return out

    # Mask detailed geometry and local handcrafted appearance in occluded
    # regions while keeping absolute node positions for graph propagation.
    out[inside, GEOMETRY_START_INDEX:GEOMETRY_END_INDEX] = 0.0
    out[inside, APPEARANCE_START_INDEX:APPEARANCE_END_INDEX] = 0.0
    out[inside, RELIABILITY_INDEX] *= 0.25
    out[inside, OCC_FLAG_INDEX] = 1.0
    return out


class LFWOccludedGraphDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleMeta],
        image_size: int,
        num_nodes: int,
        knn_k: int,
        graph_mode: str,
        rgb_window_size: int,
        cache_path: Path,
        train_mode: bool,
        occlusion_prob: float,
        record_root: Optional[Path] = None,
        return_image: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.num_nodes = num_nodes
        self.knn_k = knn_k
        self.graph_mode = graph_mode
        self.rgb_window_size = rgb_window_size
        self.train_mode = train_mode
        self.occlusion_prob = occlusion_prob
        self.return_image = return_image
        self.cache_path = cache_path
        self.extractor = LandmarkExtractor(num_nodes=num_nodes)
        self.record_reader = MXIndexedRecordReader(record_root) if record_root is not None else None
        self.invalid_paths = set()

        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        if cache_path.exists():
            raw = np.load(str(cache_path), allow_pickle=True)
            self.cache = raw["cache"].item()
            self.cache = {
                key: entry
                for key, entry in self.cache.items()
                if entry["node_feat"].shape == (self.num_nodes, NODE_FEAT_DIM)
                and entry["adj"].shape == (self.num_nodes, self.num_nodes)
            }

        self._warm_cache_if_needed()

    def _warm_cache_if_needed(self) -> None:
        updated = False
        valid_samples: List[SampleMeta] = []
        for s in tqdm(self.samples, desc="Caching landmarks", leave=False):
            key = s.image_path
            if key in self.cache:
                valid_samples.append(s)
                continue

            img = self._read_image(s)
            if img is None:
                self.invalid_paths.add(key)
                continue
            img = cv2.resize(img, (self.image_size, self.image_size))
            coords = self.extractor.extract(img)
            if np.all(coords[:, 0] < 0) or np.all(coords[:, 1] < 0):
                self.invalid_paths.add(key)
                continue
            feat = make_node_features(img, coords, self.rgb_window_size)
            adj = build_graph_adjacency(coords, self.knn_k, self.graph_mode)
            self.cache[key] = {
                "node_feat": feat.astype(np.float32),
                "adj": adj.astype(np.float32),
            }
            valid_samples.append(s)
            updated = True

        if updated:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(self.cache_path), cache=self.cache)

        self.samples = valid_samples

    def _read_image(self, sample: SampleMeta) -> Optional[np.ndarray]:
        if sample.image_path.startswith(MXREC_PREFIX):
            if self.record_reader is None:
                return None
            record_key = int(sample.image_path[len(MXREC_PREFIX):])
            return self.record_reader.read_image(record_key)
        return cv2.imread(sample.image_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        entry = self.cache.get(s.image_path)
        if entry is None:
            raise KeyError(f"Missing cached face graph for sample: {s.image_path}")
        else:
            node_feat = entry["node_feat"].copy()
            adj = entry["adj"].copy()

        if not self.return_image:
            return (
                torch.from_numpy(node_feat),
                torch.from_numpy(adj),
                torch.tensor(s.label, dtype=torch.long),
            )

        img = self._read_image(s)
        if img is None:
            image_tensor = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        return (
            torch.from_numpy(node_feat),
            torch.from_numpy(adj),
            image_tensor,
            torch.tensor(s.label, dtype=torch.long),
        )


class BalancedIdentityBatchSampler(Sampler[List[int]]):
    def __init__(self, samples: Sequence[SampleMeta], batch_size: int, instances_per_identity: int = 2):
        if instances_per_identity < 2:
            raise ValueError("instances_per_identity must be at least 2 for metric learning.")
        self.batch_size = batch_size
        self.instances_per_identity = instances_per_identity
        self.labels_to_indices: Dict[int, List[int]] = {}
        for idx, sample in enumerate(samples):
            self.labels_to_indices.setdefault(sample.label, []).append(idx)
        self.valid_labels = [label for label, indices in self.labels_to_indices.items() if len(indices) >= instances_per_identity]
        if not self.valid_labels:
            raise ValueError("BalancedIdentityBatchSampler requires at least one identity with multiple samples.")
        self.identities_per_batch = max(1, batch_size // instances_per_identity)
        self.num_batches = max(1, len(samples) // batch_size)

    def __iter__(self):
        rng = random.Random()
        for _ in range(self.num_batches):
            selected_labels = rng.sample(self.valid_labels, k=min(self.identities_per_batch, len(self.valid_labels)))
            batch: List[int] = []
            for label in selected_labels:
                indices = self.labels_to_indices[label]
                if len(indices) >= self.instances_per_identity:
                    chosen = rng.sample(indices, self.instances_per_identity)
                else:
                    chosen = rng.choices(indices, k=self.instances_per_identity)
                batch.extend(chosen)

            while len(batch) < self.batch_size:
                label = rng.choice(self.valid_labels)
                indices = self.labels_to_indices[label]
                batch.append(rng.choice(indices))

            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.num_batches


def compute_class_weights(samples: Sequence[SampleMeta], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for sample in samples:
        counts[sample.label] += 1.0

    counts = np.maximum(counts, 1.0)
    weights = np.sum(counts) / (num_classes * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(samples: Sequence[SampleMeta], num_classes: int) -> WeightedRandomSampler:
    class_weights = compute_class_weights(samples, num_classes).numpy()
    sample_weights = [float(class_weights[s.label]) for s in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


class MaskedGraphAttentionConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim)
        self.out = nn.Linear(out_dim, out_dim)
        self.scale = out_dim ** -0.5

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        score = torch.matmul(q, k.transpose(1, 2)) * self.scale
        score = score + torch.log(adj_norm.clamp(min=1e-6))
        score = score.masked_fill(adj_norm <= 0, -1e4)
        alpha = torch.softmax(score, dim=-1)
        return self.out(alpha @ v)


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return adj_norm @ self.linear(x)


class NodeWiseMLPConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        _ = adj_norm
        return self.linear(x)


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        mid_dim = max(64, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class ResidualGCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, conv_mode: str, message_scale_init: float):
        super().__init__()
        if conv_mode == "gat":
            self.conv = MaskedGraphAttentionConv(hidden_dim, hidden_dim)
        elif conv_mode == "gcn":
            self.conv = DenseGraphConv(hidden_dim, hidden_dim)
        elif conv_mode == "mlp":
            self.conv = NodeWiseMLPConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported conv_mode: {conv_mode}")
        self.message_scale = nn.Parameter(torch.tensor(float(message_scale_init)))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, adj_norm)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        scale = torch.clamp(self.message_scale, min=0.0, max=1.0)
        return x + scale * h


class OcclusionAwareInputGate(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, occ_flag: torch.Tensor, return_attention: bool = False):
        gate_input = torch.cat([x, occ_flag], dim=-1)
        learned_gate = torch.sigmoid(self.gate(gate_input))
        prior_gate = 1.0 - 0.8 * occ_flag
        alpha = learned_gate * prior_gate
        out = alpha * x
        if return_attention:
            return out, alpha
        return out


class OcclusionAwareNodeAttention(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, occ_flag: torch.Tensor, return_attention: bool = False):
        gate_input = torch.cat([x, occ_flag], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))  # [B, N, 1]
        out = alpha * x
        if return_attention:
            return out, alpha
        return out


class RegionAttentionPool(nn.Module):
    def __init__(self, in_dim: int, num_nodes: int, hidden: int = 64):
        super().__init__()
        self.node_scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.region_scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("region_masks", build_region_masks(num_nodes), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        valid_flag: torch.Tensor | None = None,
        reliability_flag: torch.Tensor | None = None,
        occ_flag: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        score = self.node_scorer(x)
        if valid_flag is not None:
            score = score.masked_fill(valid_flag <= 0.5, -1e4)
        if reliability_flag is not None:
            score = score + torch.log(reliability_flag.clamp(min=1e-4))
        if occ_flag is not None:
            score = score - 2.0 * occ_flag
        region_features = []
        region_node_alphas = []

        for region_mask in self.region_masks:
            mask = region_mask.view(1, -1, 1)
            masked_score = score.masked_fill(~mask, -1e4)
            alpha = torch.softmax(masked_score, dim=1)
            region_feat = torch.sum(alpha * x, dim=1)
            region_features.append(region_feat)
            region_node_alphas.append(alpha)

        regions = torch.stack(region_features, dim=1)
        region_score = self.region_scorer(regions)
        region_alpha = torch.softmax(region_score, dim=1)
        out = torch.sum(region_alpha * regions, dim=1)

        if return_attention:
            node_alpha = torch.zeros_like(score)
            for idx, alpha in enumerate(region_node_alphas):
                node_alpha = node_alpha + region_alpha[:, idx:idx + 1, :] * alpha
            return out, node_alpha
        return out


class MeanPool(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        valid_flag: torch.Tensor | None = None,
        reliability_flag: torch.Tensor | None = None,
        occ_flag: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        _ = valid_flag, reliability_flag, occ_flag
        out = torch.mean(x, dim=1)
        if return_attention:
            alpha = torch.full(
                (x.size(0), x.size(1), 1),
                1.0 / max(x.size(1), 1),
                dtype=x.dtype,
                device=x.device,
            )
            return out, alpha
        return out


class NodeAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_flag: torch.Tensor | None = None,
        reliability_flag: torch.Tensor | None = None,
        occ_flag: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        score = self.scorer(x)
        if valid_flag is not None:
            score = score.masked_fill(valid_flag <= 0.5, -1e4)
        if reliability_flag is not None:
            score = score + torch.log(reliability_flag.clamp(min=1e-4))
        if occ_flag is not None:
            score = score - 2.0 * occ_flag
        alpha = torch.softmax(score, dim=1)
        out = torch.sum(alpha * x, dim=1)
        if return_attention:
            return out, alpha
        return out


class SemanticRegionAggregator(nn.Module):
    def __init__(self, num_nodes: int):
        super().__init__()
        self.register_buffer("region_masks", build_semantic_region_masks(num_nodes), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        valid_flag: torch.Tensor,
        reliability_flag: torch.Tensor,
        occ_flag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        region_features = []
        region_valid_ratios = []
        region_reliabilities = []
        region_occ_ratios = []

        node_weight = valid_flag * reliability_flag * (1.0 - 0.5 * occ_flag)
        for region_mask in self.region_masks:
            mask = region_mask.view(1, -1, 1).to(x.device)
            masked_weight = node_weight * mask
            denom = masked_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
            region_feat = (masked_weight * x).sum(dim=1) / denom.squeeze(1)
            region_features.append(region_feat)

            region_size = mask.sum(dim=1).clamp(min=1.0)
            valid_ratio = (valid_flag * mask).sum(dim=1) / region_size
            reliability = (reliability_flag * valid_flag * mask).sum(dim=1) / (valid_flag * mask).sum(dim=1).clamp(min=1e-6)
            occ_ratio = (occ_flag * valid_flag * mask).sum(dim=1) / (valid_flag * mask).sum(dim=1).clamp(min=1e-6)
            region_valid_ratios.append(valid_ratio)
            region_reliabilities.append(reliability)
            region_occ_ratios.append(occ_ratio)

        region_x = torch.stack(region_features, dim=1)
        region_valid = torch.stack(region_valid_ratios, dim=1)
        region_reliability = torch.stack(region_reliabilities, dim=1)
        region_occ = torch.stack(region_occ_ratios, dim=1)
        return region_x, region_valid, region_reliability, region_occ


class RegionGraphAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_flag: torch.Tensor,
        reliability_flag: torch.Tensor,
        occ_flag: torch.Tensor,
        return_attention: bool = False,
    ):
        score_input = torch.cat([x, valid_flag, reliability_flag, occ_flag], dim=-1)
        score = self.scorer(score_input)
        score = score.masked_fill(valid_flag <= 0.0, -1e4)
        score = score + torch.log(reliability_flag.clamp(min=1e-4))
        score = score - 2.5 * occ_flag
        alpha = torch.softmax(score, dim=1)
        out = torch.sum(alpha * x, dim=1)
        if return_attention:
            return out, alpha
        return out


class LandmarkCNNFeatureEncoder(nn.Module):
    def __init__(self, out_dim: int, dropout: float, backbone: str):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported CNN backbone: {backbone}")
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(image)
        grid = coords_xy.clamp(0.0, 1.0) * 2.0 - 1.0
        grid = grid.view(grid.size(0), grid.size(1), 1, 2)
        sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=True)
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return self.proj(sampled)


class OccludedFaceGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        num_nodes: int,
        conv_modes: Sequence[str],
        pool_mode: str,
        use_input_gate: bool,
        use_node_attention: bool,
        message_scale_init: float,
        jk_mode: str,
        cnn_branch: str,
        cnn_backbone: str,
        cnn_dropout: float,
    ):
        super().__init__()
        self.use_input_gate = use_input_gate
        self.use_node_attention = use_node_attention
        self.jk_mode = jk_mode
        self.region_count = len(REGION_NAMES)
        self.encoder = NodeEncoder(in_dim, hidden_dim)
        self.input_gate = OcclusionAwareInputGate(hidden_dim)
        self.region_aggregator = SemanticRegionAggregator(num_nodes)
        self.register_buffer("region_adj", build_region_adjacency(), persistent=False)
        if not conv_modes:
            raise ValueError("conv_modes must contain at least one graph layer.")
        self.conv_modes = list(conv_modes)
        self.gcn_layers = nn.ModuleList(
            [ResidualGCNBlock(hidden_dim, dropout, mode, message_scale_init) for mode in self.conv_modes]
        )
        if jk_mode == "concat":
            self.jk_fuse = nn.Sequential(
                nn.Linear(hidden_dim * (len(self.gcn_layers) + 1), hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
            )
        elif jk_mode == "last":
            self.jk_fuse = nn.Identity()
        else:
            raise ValueError(f"Unsupported jk_mode: {jk_mode}")
        self.node_attention = OcclusionAwareNodeAttention(hidden_dim)
        if pool_mode == "mean":
            self.pool = MeanPool()
        elif pool_mode == "node_attention":
            self.pool = NodeAttentionPool(hidden_dim)
        elif pool_mode == "region_attention":
            self.pool = RegionGraphAttentionPool(hidden_dim)
        else:
            raise ValueError(f"Unsupported pool_mode: {pool_mode}")
        self.cnn_branch = cnn_branch
        if cnn_branch == "none":
            self.cnn_encoder = None
            self.cnn_node_gate = None
        elif cnn_branch == "node":
            self.cnn_encoder = LandmarkCNNFeatureEncoder(hidden_dim, dropout=cnn_dropout, backbone=cnn_backbone)
            self.cnn_node_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported cnn_branch: {cnn_branch}")
        self.head = nn.Linear(hidden_dim, num_classes)
        self.last_input_gate = None
        self.last_attention = None
        self.last_node_attention = None

    def build_region_graph(
        self,
        region_valid: torch.Tensor,
        region_reliability: torch.Tensor,
        region_occ: torch.Tensor,
    ) -> torch.Tensor:
        base_adj = self.region_adj.unsqueeze(0).to(region_valid.device)
        source_weight = region_valid.transpose(1, 2) * region_reliability.transpose(1, 2)
        target_weight = region_valid * region_reliability
        occ_weight = 1.0 - 0.7 * region_occ.transpose(1, 2)
        adj = base_adj * source_weight * target_weight * occ_weight
        eye = torch.eye(self.region_count, dtype=adj.dtype, device=adj.device).unsqueeze(0)
        adj = torch.maximum(adj, eye)
        return normalize_adjacency(adj)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        image: torch.Tensor | None = None,
        return_attention: bool = False,
        return_features: bool = False,
    ):
        coords_xy = x[:, :, 0:2]
        reliability_flag = x[:, :, RELIABILITY_INDEX:RELIABILITY_INDEX + 1]
        valid_flag = x[:, :, VALID_FLAG_INDEX:VALID_FLAG_INDEX + 1]
        occ_flag = x[:, :, OCC_FLAG_INDEX:OCC_FLAG_INDEX + 1]

        x = self.encoder(x)
        if self.cnn_encoder is not None:
            if image is None:
                raise ValueError("CNN node branch requires image tensors from the dataset.")
            cnn_node_feat = self.cnn_encoder(image, coords_xy)
            cnn_gate = self.cnn_node_gate(torch.cat([x, cnn_node_feat], dim=-1))
            x = x + cnn_gate * cnn_node_feat
        if self.use_input_gate and return_attention:
            x, input_alpha = self.input_gate(x, occ_flag, return_attention=True)
        elif self.use_input_gate:
            x = self.input_gate(x, occ_flag)
            input_alpha = None
        else:
            input_alpha = None

        region_x, region_valid, region_reliability, region_occ = self.region_aggregator(
            x,
            valid_flag=valid_flag,
            reliability_flag=reliability_flag,
            occ_flag=occ_flag,
        )
        region_adj_norm = self.build_region_graph(region_valid, region_reliability, region_occ)

        layer_outputs = [region_x]
        for layer in self.gcn_layers:
            layer_outputs.append(layer(layer_outputs[-1], region_adj_norm))
        if self.jk_mode == "concat":
            x = self.jk_fuse(torch.cat(layer_outputs, dim=-1))
        else:
            x = layer_outputs[-1]

        if self.use_node_attention and return_attention:
            x, node_alpha = self.node_attention(x, region_occ, return_attention=True)
        elif self.use_node_attention:
            x = self.node_attention(x, region_occ)
            node_alpha = None
        else:
            node_alpha = None

        if return_attention:
            g, graph_alpha = self.pool(
                x,
                valid_flag=region_valid,
                reliability_flag=region_reliability,
                occ_flag=region_occ,
                return_attention=True,
            )
            self.last_input_gate = input_alpha
            self.last_node_attention = node_alpha
            self.last_attention = graph_alpha
            logits = self.head(g)
            if return_features:
                return logits, g, {
                    "input_gate": input_alpha,
                    "node_attention": node_alpha,
                    "graph_attention": graph_alpha,
            }
            return logits, {
                "input_gate": input_alpha,
                "node_attention": node_alpha,
                "graph_attention": graph_alpha,
            }

        g = self.pool(x, valid_flag=region_valid, reliability_flag=region_reliability, occ_flag=region_occ)
        self.last_input_gate = None
        self.last_attention = None
        self.last_node_attention = None
        logits = self.head(g)
        if return_features:
            return logits, g
        return logits


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    if embeddings.size(0) < 2:
        return embeddings.new_tensor(0.0)

    emb = F.normalize(embeddings, p=2, dim=1)
    dist = torch.cdist(emb, emb, p=2)
    labels = labels.view(-1)
    same = labels[:, None] == labels[None, :]
    diff = ~same
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    same = same & ~eye

    has_pos = same.any(dim=1)
    has_neg = diff.any(dim=1)
    valid = has_pos & has_neg
    if not torch.any(valid):
        return embeddings.new_tensor(0.0)

    hardest_pos = dist.masked_fill(~same, float("-inf")).max(dim=1).values
    hardest_neg = dist.masked_fill(~diff, float("inf")).min(dim=1).values
    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss[valid].mean()


def occlusion_consistency_loss(clean_feat: torch.Tensor, occ_feat: torch.Tensor) -> torch.Tensor:
    clean_feat = F.normalize(clean_feat, p=2, dim=1)
    occ_feat = F.normalize(occ_feat, p=2, dim=1)
    return F.mse_loss(clean_feat, occ_feat)


def unpack_batch(batch, device: torch.device):
    if len(batch) == 4:
        node_feat, adj, image, y = batch
        image = image.to(device)
    else:
        node_feat, adj, y = batch
        image = None
    return node_feat.to(device), adj.to(device), image, y.to(device)


@torch.no_grad()
def build_class_prototypes(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[int, torch.Tensor]:
    model.eval()
    by_class: Dict[int, List[torch.Tensor]] = {}

    for batch in loader:
        node_feat, adj, image, y = unpack_batch(batch, device)
        _, graph_feat = model(node_feat, adj, image=image, return_features=True)
        graph_feat = F.normalize(graph_feat, p=2, dim=1)

        for feat, label in zip(graph_feat, y):
            by_class.setdefault(int(label.item()), []).append(feat.detach().cpu())

    prototypes: Dict[int, torch.Tensor] = {}
    for label, feats in by_class.items():
        stacked = torch.stack(feats, dim=0)
        proto = stacked.mean(dim=0)
        prototypes[label] = F.normalize(proto, p=2, dim=0)

    return prototypes


@torch.no_grad()
def evaluate(model: nn.Module, gallery_loader: DataLoader, probe_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0

    all_pred = []
    all_true = []
    prototypes = build_class_prototypes(model, gallery_loader, device)
    sorted_labels = sorted(prototypes.keys())
    proto_mat = torch.stack([prototypes[label] for label in sorted_labels], dim=0).to(device)

    for batch in probe_loader:
        node_feat, adj, image, y = unpack_batch(batch, device)
        _, graph_feat = model(node_feat, adj, image=image, return_features=True)
        graph_feat = F.normalize(graph_feat, p=2, dim=1)
        scores = graph_feat @ proto_mat.t()
        pred_idx = torch.argmax(scores, dim=1)
        pred = torch.tensor([sorted_labels[idx] for idx in pred_idx.tolist()], device=device)

        total += y.size(0)
        correct += (pred == y).sum().item()
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    acc = correct / max(total, 1)
    macro_f1 = macro_f1_score(np.concatenate(all_true), np.concatenate(all_pred))
    return acc, macro_f1


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    all_pred = []
    all_true = []

    for batch in loader:
        node_feat, adj, image, y = unpack_batch(batch, device)
        logits = model(node_feat, adj, image=image)
        pred = torch.argmax(logits, dim=1)

        total += y.size(0)
        correct += (pred == y).sum().item()
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    acc = correct / max(total, 1)
    macro_f1 = macro_f1_score(np.concatenate(all_true), np.concatenate(all_pred))
    return acc, macro_f1


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    f1_list = []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_list.append(f1)

    return float(np.mean(f1_list)) if f1_list else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    warmup_epochs: int,
    occlusion_prob: float,
    metric_loss: str,
    metric_weight: float,
    triplet_margin: float,
    consistency_weight: float,
    grad_clip: float,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    if warmup_epochs > 0:
        if epoch <= warmup_epochs:
            schedule = 0.0
        else:
            schedule = min(1.0, (epoch - warmup_epochs) / max(warmup_epochs, 1))
    else:
        schedule = 1.0

    effective_occlusion_prob = occlusion_prob * schedule
    effective_metric_weight = metric_weight * schedule
    effective_consistency_weight = consistency_weight * schedule

    for batch in tqdm(loader, desc="Train", leave=False):
        node_feat, adj, image, y = unpack_batch(batch, device)
        clean_node_feat = node_feat.clone()

        optimizer.zero_grad()
        logits_clean, graph_feat_clean = model(clean_node_feat, adj, image=image, return_features=True)
        loss = criterion(logits_clean, y)

        if schedule > 0.0:
            occ_node_feat = torch.stack(
                [torch.from_numpy(apply_random_occlusion(sample.detach().cpu().numpy(), effective_occlusion_prob)) for sample in node_feat],
                dim=0,
            ).to(device)
            logits_occ, graph_feat_occ = model(occ_node_feat, adj, image=image, return_features=True)
            cls_loss = 0.5 * (criterion(logits_clean, y) + criterion(logits_occ, y))
            cons_loss = occlusion_consistency_loss(graph_feat_clean, graph_feat_occ)
            loss = cls_loss + effective_consistency_weight * cons_loss

            if metric_loss == "triplet" and effective_metric_weight > 0:
                pair_feat = torch.cat([graph_feat_clean, graph_feat_occ], dim=0)
                pair_labels = torch.cat([y, y], dim=0)
                tri_loss = batch_hard_triplet_loss(pair_feat, pair_labels, margin=triplet_margin)
                loss = loss + effective_metric_weight * tri_loss

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_correct += (torch.argmax(logits_clean.detach(), dim=1) == y).sum().item()
        running_total += y.size(0)

    return running_loss / max(running_total, 1), running_correct / max(running_total, 1), schedule


def parse_args():
    p = argparse.ArgumentParser(description="Occluded Face Recognition with GCN + Attention on LFW")
    p.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "lfw-deepfunneled" / "lfw-deepfunneled"))
    p.add_argument("--save-dir", type=str, default=str(PROJECT_ROOT / "runs" / "occluded_gcn_lab_cnn_lfw"))
    p.add_argument("--cache-path", type=str, default="", help="Optional shared landmark cache path. Defaults to save-dir cache.")
    p.add_argument("--dataset-format", type=str, default="auto", choices=["auto", "lfw_dir", "mxrec"])
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--num-nodes", type=int, default=468)
    p.add_argument("--knn-k", type=int, default=6)
    p.add_argument("--graph-mode", type=str, default="topology_knn", choices=["self", "knn", "topology", "topology_knn"])
    p.add_argument(
        "--conv-modes",
        type=str,
        default="gcn,gcn,gat,gat,gat",
        help="Comma-separated graph block types, e.g. gcn,gcn,gat,gat,gat.",
    )
    p.add_argument("--pool-mode", type=str, default="region_attention", choices=["mean", "node_attention", "region_attention"])
    p.add_argument("--use-input-gate", action="store_true")
    p.add_argument("--use-node-attention", action="store_true")
    p.add_argument("--message-scale-init", type=float, default=0.5)
    p.add_argument("--jk-mode", type=str, default="concat", choices=["last", "concat"])
    p.add_argument("--rgb-window-size", type=int, default=7, help="Local window size for handcrafted node appearance descriptors.")
    p.add_argument("--min-images-per-identity", type=int, default=10)
    # Start with an easier closed-set setting to verify the model can learn
    # before scaling to larger class counts.
    p.add_argument("--max-classes", type=int, default=30)
    p.add_argument("--max-images-per-class", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--occlusion-prob", type=float, default=0.3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--cnn-branch", type=str, default="none", choices=["none", "node"])
    p.add_argument("--cnn-backbone", type=str, default="resnet18", choices=["resnet18"])
    p.add_argument("--cnn-dropout", type=float, default=0.2)
    p.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--class-weighting", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metric-loss", type=str, default="triplet", choices=["none", "triplet"])
    p.add_argument("--metric-weight", type=float, default=0.2)
    p.add_argument("--triplet-margin", type=float, default=0.3)
    p.add_argument("--consistency-weight", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    dataset_format = args.dataset_format
    if dataset_format == "auto":
        dataset_format = "mxrec" if (data_root / "train.rec").exists() and (data_root / "train.idx").exists() else "lfw_dir"

    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = save_dir / (
            f"landmark_cache_{dataset_format}_{args.num_nodes}n_feat{NODE_FEAT_DIM}_local{args.rgb_window_size}"
            f"_{args.graph_mode}.npz"
        )
    record_root = data_root if dataset_format == "mxrec" else None

    if dataset_format == "mxrec":
        samples, class_to_idx = scan_mxrec(
            root=data_root,
            min_images_per_identity=args.min_images_per_identity,
            max_classes=args.max_classes,
            max_images_per_class=args.max_images_per_class,
        )
    else:
        samples, class_to_idx = scan_lfw(
            root=data_root,
            min_images_per_identity=args.min_images_per_identity,
            max_classes=args.max_classes,
            max_images_per_class=args.max_images_per_class,
        )

    if len(class_to_idx) < 2:
        raise RuntimeError("Valid classes < 2. Lower --min-images-per-identity or check dataset path.")

    conv_modes = [mode.strip().lower() for mode in args.conv_modes.split(",") if mode.strip()]
    valid_conv_modes = {"mlp", "gcn", "gat"}
    invalid_conv_modes = [mode for mode in conv_modes if mode not in valid_conv_modes]
    if not conv_modes:
        raise ValueError("--conv-modes must contain at least one layer type.")
    if invalid_conv_modes:
        raise ValueError(f"Unsupported conv modes: {invalid_conv_modes}. Valid modes: {sorted(valid_conv_modes)}")

    train_s, val_s, test_s = split_by_identity(samples, args.train_ratio, args.val_ratio, args.seed)
    return_image = args.cnn_branch != "none"

    train_ds = LFWOccludedGraphDataset(
        samples=train_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        graph_mode=args.graph_mode,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=True,
        occlusion_prob=args.occlusion_prob,
        record_root=record_root,
        return_image=return_image,
    )
    train_eval_ds = LFWOccludedGraphDataset(
        samples=train_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        graph_mode=args.graph_mode,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=False,
        occlusion_prob=0.0,
        record_root=record_root,
        return_image=return_image,
    )
    val_ds = LFWOccludedGraphDataset(
        samples=val_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        graph_mode=args.graph_mode,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=False,
        occlusion_prob=0.0,
        record_root=record_root,
        return_image=return_image,
    )
    test_ds = LFWOccludedGraphDataset(
        samples=test_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        graph_mode=args.graph_mode,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=False,
        occlusion_prob=0.0,
        record_root=record_root,
        return_image=return_image,
    )

    if args.metric_loss != "none":
        train_sampler = BalancedIdentityBatchSampler(train_ds.samples, batch_size=args.batch_size, instances_per_identity=2)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers)
    elif args.balanced_sampling:
        train_sampler = build_weighted_sampler(train_ds.samples, len(class_to_idx))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OccludedFaceGCN(
        in_dim=NODE_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        num_classes=len(class_to_idx),
        dropout=args.dropout,
        num_nodes=args.num_nodes,
        conv_modes=conv_modes,
        pool_mode=args.pool_mode,
        use_input_gate=args.use_input_gate,
        use_node_attention=args.use_node_attention,
        message_scale_init=args.message_scale_init,
        jk_mode=args.jk_mode,
        cnn_branch=args.cnn_branch,
        cnn_backbone=args.cnn_backbone,
        cnn_dropout=args.cnn_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(train_ds.samples, len(class_to_idx)).to(device) if args.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_val_acc = -1.0
    history = []

    print(f"Device: {device}")
    print(f"Dataset: {dataset_format} | root={data_root}")
    print(f"Classes: {len(class_to_idx)} | Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(
        f"Ablation config: graph={args.graph_mode} | conv_layers={','.join(conv_modes)} | pool={args.pool_mode} "
        f"| input_gate={args.use_input_gate} | node_attention={args.use_node_attention} "
        f"| message_scale_init={args.message_scale_init} | jk={args.jk_mode}"
    )
    print(f"Class balance: weighted_loss={args.class_weighting} | balanced_sampling={args.balanced_sampling}")
    print(f"CNN branch: {args.cnn_branch} | cnn_backbone={args.cnn_backbone} | cnn_dropout={args.cnn_dropout}")
    print("Evaluation mode: feature matching against train-set class prototypes")
    print("Input features: landmark geometry + handcrafted local appearance descriptors + landmark reliability")
    print("Backbone mode: landmark local encoding -> semantic region graph propagation -> reliability-aware region pooling")
    print("Training mode: clean warmup, then region-graph occlusion optimization")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, occ_schedule = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            occlusion_prob=args.occlusion_prob,
            metric_loss=args.metric_loss,
            metric_weight=args.metric_weight,
            triplet_margin=args.triplet_margin,
            consistency_weight=args.consistency_weight,
            grad_clip=args.grad_clip,
        )
        val_acc, val_f1 = evaluate(model, train_eval_loader, val_loader, device)
        val_cls_acc, val_cls_f1 = evaluate_classifier(model, val_loader, device)

        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "occlusion_schedule": occ_schedule,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "val_cls_acc": val_cls_acc,
            "val_cls_macro_f1": val_cls_f1,
        }
        history.append(item)
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | train_acc={train_acc:.4f} "
            f"| occ={occ_schedule:.2f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} "
            f"| val_cls_acc={val_cls_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")

    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    test_acc, test_f1 = evaluate(model, train_eval_loader, test_loader, device)
    print(f"[Best] test_acc={test_acc:.4f} | test_macro_f1={test_f1:.4f}")

    with open(save_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "test_macro_f1": test_f1,
                "num_classes": len(class_to_idx),
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "test_size": len(test_ds),
                "dataset_format": dataset_format,
                "cnn_branch": args.cnn_branch,
                "cnn_backbone": args.cnn_backbone,
                "cnn_dropout": args.cnn_dropout,
                "region_count": len(REGION_NAMES),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
