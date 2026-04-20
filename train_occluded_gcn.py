import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

NODE_FEAT_DIM = 10
OCC_FLAG_INDEX = 9


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

        model_path = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
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


def build_knn_adjacency(coords_xy: np.ndarray, k: int) -> np.ndarray:
    n = coords_xy.shape[0]
    valid = np.where(coords_xy[:, 0] >= 0)[0]
    adj = np.zeros((n, n), dtype=np.float32)

    if len(valid) <= 1:
        np.fill_diagonal(adj, 1.0)
        return adj

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


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    deg = torch.sum(adj, dim=-1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    d = torch.diag_embed(deg_inv_sqrt)
    return d @ adj @ d


def make_node_features(image_bgr: np.ndarray, coords_xy: np.ndarray, rgb_window_size: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    n = coords_xy.shape[0]
    feat = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
    half_window = max(0, rgb_window_size // 2)

    for i, (x, y) in enumerate(coords_xy):
        if x < 0 or y < 0:
            feat[i] = 0.0
            feat[i, OCC_FLAG_INDEX] = 0.0
            continue

        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        x0 = max(0, xi - half_window)
        x1 = min(w, xi + half_window + 1)
        y0 = max(0, yi - half_window)
        y1 = min(h, yi + half_window + 1)
        patch_rgb = image_rgb[y0:y1, x0:x1]
        mean_rgb = np.mean(patch_rgb, axis=(0, 1))
        std_rgb = np.std(patch_rgb, axis=(0, 1))
        mean_grad = float(np.mean(grad_mag[y0:y1, x0:x1]))
        feat[i, 0] = x / max(w - 1, 1)
        feat[i, 1] = y / max(h - 1, 1)
        feat[i, 2:5] = mean_rgb
        feat[i, 5:8] = std_rgb
        feat[i, 8] = mean_grad
        feat[i, OCC_FLAG_INDEX] = 0.0

    return feat


def sample_occlusion_mask(node_feat: np.ndarray) -> np.ndarray:
    x = node_feat[:, 0]
    y = node_feat[:, 1]
    valid = node_feat[:, OCC_FLAG_INDEX] < 0.5
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

    # Suppress local appearance features while retaining geometric coordinates.
    out[inside, 2:9] = 0.0
    out[inside, OCC_FLAG_INDEX] = 1.0
    return out


class LFWOccludedGraphDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleMeta],
        image_size: int,
        num_nodes: int,
        knn_k: int,
        rgb_window_size: int,
        cache_path: Path,
        train_mode: bool,
        occlusion_prob: float,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.num_nodes = num_nodes
        self.knn_k = knn_k
        self.rgb_window_size = rgb_window_size
        self.train_mode = train_mode
        self.occlusion_prob = occlusion_prob
        self.cache_path = cache_path
        self.extractor = LandmarkExtractor(num_nodes=num_nodes)
        self.invalid_paths = set()

        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        if cache_path.exists():
            raw = np.load(str(cache_path), allow_pickle=True)
            self.cache = raw["cache"].item()

        self._warm_cache_if_needed()

    def _warm_cache_if_needed(self) -> None:
        updated = False
        valid_samples: List[SampleMeta] = []
        for s in tqdm(self.samples, desc="Caching landmarks", leave=False):
            key = s.image_path
            if key in self.cache:
                valid_samples.append(s)
                continue

            img = cv2.imread(s.image_path)
            if img is None:
                self.invalid_paths.add(key)
                continue
            img = cv2.resize(img, (self.image_size, self.image_size))
            coords = self.extractor.extract(img)
            if np.all(coords[:, 0] < 0) or np.all(coords[:, 1] < 0):
                self.invalid_paths.add(key)
                continue
            feat = make_node_features(img, coords, self.rgb_window_size)
            adj = build_knn_adjacency(coords, self.knn_k)
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

        return (
            torch.from_numpy(node_feat),
            torch.from_numpy(adj),
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


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return adj_norm @ h


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
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv = GraphConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, adj_norm)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return x + h


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


class NodeAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        score = self.scorer(x)  # [B, N, 1]
        alpha = torch.softmax(score, dim=1)
        out = torch.sum(alpha * x, dim=1)
        if return_attention:
            return out, alpha
        return out


class OccludedFaceGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = NodeEncoder(in_dim, hidden_dim)
        self.input_gate = OcclusionAwareInputGate(hidden_dim)
        self.gcn1 = ResidualGCNBlock(hidden_dim, dropout)
        self.gcn2 = ResidualGCNBlock(hidden_dim, dropout)
        self.gcn3 = ResidualGCNBlock(hidden_dim, dropout)
        self.node_attention = OcclusionAwareNodeAttention(hidden_dim)
        self.pool = NodeAttentionPool(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.last_input_gate = None
        self.last_attention = None
        self.last_node_attention = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor, return_attention: bool = False, return_features: bool = False):
        # Normalize the graph once per forward pass to match the standard
        # propagation rule H^(l+1)=sigma(A_hat H^(l) W^(l)).
        occ_flag = x[:, :, OCC_FLAG_INDEX:OCC_FLAG_INDEX + 1]
        adj_norm = normalize_adjacency(adj)

        x = self.encoder(x)
        if return_attention:
            x, input_alpha = self.input_gate(x, occ_flag, return_attention=True)
        else:
            x = self.input_gate(x, occ_flag)
            input_alpha = None
        x = self.gcn1(x, adj_norm)
        x = self.gcn2(x, adj_norm)
        x = self.gcn3(x, adj_norm)
        if return_attention:
            x, node_alpha = self.node_attention(x, occ_flag, return_attention=True)
        else:
            x = self.node_attention(x, occ_flag)
            node_alpha = None

        if return_attention:
            g, graph_alpha = self.pool(x, return_attention=True)
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

        g = self.pool(x)
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


@torch.no_grad()
def build_class_prototypes(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[int, torch.Tensor]:
    model.eval()
    by_class: Dict[int, List[torch.Tensor]] = {}

    for node_feat, adj, y in loader:
        node_feat = node_feat.to(device)
        adj = adj.to(device)
        y = y.to(device)
        _, graph_feat = model(node_feat, adj, return_features=True)
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

    for node_feat, adj, y in probe_loader:
        node_feat = node_feat.to(device)
        adj = adj.to(device)
        y = y.to(device)
        _, graph_feat = model(node_feat, adj, return_features=True)
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
    occlusion_prob: float,
    metric_loss: str,
    metric_weight: float,
    triplet_margin: float,
    consistency_weight: float,
) -> float:
    model.train()
    running_loss = 0.0

    for node_feat, adj, y in tqdm(loader, desc="Train", leave=False):
        clean_node_feat = node_feat.clone()
        occ_node_feat = torch.stack(
            [torch.from_numpy(apply_random_occlusion(sample.numpy(), occlusion_prob)) for sample in clean_node_feat],
            dim=0,
        )

        clean_node_feat = clean_node_feat.to(device)
        occ_node_feat = occ_node_feat.to(device)
        adj = adj.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits_clean, graph_feat_clean = model(clean_node_feat, adj, return_features=True)
        logits_occ, graph_feat_occ = model(occ_node_feat, adj, return_features=True)
        cls_loss = 0.5 * (criterion(logits_clean, y) + criterion(logits_occ, y))
        cons_loss = occlusion_consistency_loss(graph_feat_clean, graph_feat_occ)
        loss = cls_loss + consistency_weight * cons_loss

        if metric_loss == "triplet":
            pair_feat = torch.cat([graph_feat_clean, graph_feat_occ], dim=0)
            pair_labels = torch.cat([y, y], dim=0)
            tri_loss = batch_hard_triplet_loss(pair_feat, pair_labels, margin=triplet_margin)
            loss = loss + metric_weight * tri_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)

    return running_loss / max(len(loader.dataset), 1)


def parse_args():
    p = argparse.ArgumentParser(description="Occluded Face Recognition with GCN + Attention on LFW")
    p.add_argument("--data-root", type=str, default="data/lfw-deepfunneled/lfw-deepfunneled")
    p.add_argument("--save-dir", type=str, default="runs/occluded_gcn")
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--num-nodes", type=int, default=468)
    p.add_argument("--knn-k", type=int, default=6)
    p.add_argument("--rgb-window-size", type=int, default=3)
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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--metric-loss", type=str, default="triplet", choices=["none", "triplet"])
    p.add_argument("--metric-weight", type=float, default=0.2)
    p.add_argument("--triplet-margin", type=float, default=0.3)
    p.add_argument("--consistency-weight", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    cache_path = save_dir / f"landmark_cache_{args.num_nodes}n_feat10_rgb{args.rgb_window_size}x{args.rgb_window_size}.npz"
    save_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    samples, class_to_idx = scan_lfw(
        root=data_root,
        min_images_per_identity=args.min_images_per_identity,
        max_classes=args.max_classes,
        max_images_per_class=args.max_images_per_class,
    )

    if len(class_to_idx) < 2:
        raise RuntimeError("Valid classes < 2. Lower --min-images-per-identity or check dataset path.")

    train_s, val_s, test_s = split_by_identity(samples, args.train_ratio, args.val_ratio, args.seed)

    train_ds = LFWOccludedGraphDataset(
        samples=train_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=True,
        occlusion_prob=args.occlusion_prob,
    )
    train_eval_ds = LFWOccludedGraphDataset(
        samples=train_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=False,
        occlusion_prob=0.0,
    )
    val_ds = LFWOccludedGraphDataset(
        samples=val_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=False,
        occlusion_prob=0.0,
    )
    test_ds = LFWOccludedGraphDataset(
        samples=test_s,
        image_size=args.image_size,
        num_nodes=args.num_nodes,
        knn_k=args.knn_k,
        rgb_window_size=args.rgb_window_size,
        cache_path=cache_path,
        train_mode=False,
        occlusion_prob=0.0,
    )

    if args.metric_loss != "none":
        train_sampler = BalancedIdentityBatchSampler(train_ds.samples, batch_size=args.batch_size, instances_per_identity=2)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers)
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history = []

    print(f"Device: {device}")
    print(f"Classes: {len(class_to_idx)} | Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print("Evaluation mode: feature matching against train-set class prototypes")
    print("Training mode: clean/occluded dual-view optimization with input gating, triplet loss, and consistency loss")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            occlusion_prob=args.occlusion_prob,
            metric_loss=args.metric_loss,
            metric_weight=args.metric_weight,
            triplet_margin=args.triplet_margin,
            consistency_weight=args.consistency_weight,
        )
        val_acc, val_f1 = evaluate(model, train_eval_loader, val_loader, device)

        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(item)
        print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
