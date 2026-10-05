import os, glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SIZE = 48


def list_images(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXT))


def read_polygons(label_path, lane_class=None):
    """Read Ultralytics YOLO-seg txt: `cls x1 y1 x2 y2 ...` (normalized).
    Only polygon rows (>=3 points, even coord count) are kept. Bounding-box rows (4 coords) are ignored.
    If lane_class is given, rows of other classes are ignored."""
    polys = []
    if not os.path.isfile(label_path):
        return polys
    for line in open(label_path):
        p = line.split()
        if len(p) < 7:
            continue
        coords = p[1:]
        if len(coords) % 2:
            continue
        if lane_class is not None and int(float(p[0])) != lane_class:
            continue
        polys.append(np.array(coords, dtype=np.float32).reshape(-1, 2))
    return polys


def polygons_to_mask(polys, h, w):
    """Rasterize normalized polygons to a binary uint8 mask (0/1) of size (h, w)."""
    m = np.zeros((h, w), np.uint8)
    for poly in polys:
        pts = np.round(poly * [w, h]).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def label_path_for(img_path, labels_dir):
    return os.path.join(labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")


def augment(img):
    """img: float32 RGB in [0,1], HxWx3. Slight sunlight-like perturbations."""
    if np.random.rand() < 0.5:                       # white-balance shift (per-channel gain)
        img = img * np.random.uniform(0.92, 1.08, size=3).astype(np.float32)
    if np.random.rand() < 0.5:                       # brightness shift
        img = img + np.random.uniform(-0.08, 0.08)
    if np.random.rand() < 0.3:                       # small blur
        img = cv2.GaussianBlur(img, (3, 3), np.random.uniform(0.3, 1.0))
    return np.clip(img, 0, 1)


class LaneDataset(Dataset):
    def __init__(self, image_paths, labels_dir, lane_class=None, train=False, size=SIZE):
        self.paths, self.labels_dir = image_paths, labels_dir
        self.lane_class, self.train, self.size = lane_class, train, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)   # images have various sizes
        h, w = img.shape[:2]
        polys = read_polygons(label_path_for(self.paths[i], self.labels_dir), self.lane_class)
        mask = polygons_to_mask(polys, h, w)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img = augment(img)
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.from_numpy(mask[None].astype(np.float32))


def preprocess(img_rgb, size=SIZE):
    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return torch.from_numpy(x.transpose(2, 0, 1).copy())[None]
