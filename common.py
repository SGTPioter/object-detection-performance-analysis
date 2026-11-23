"""Funkcje wspólne: IoU, wczytywanie etykiet, dopasowanie wykryć, rysowanie."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Kolory (BGR) do rysowania.
GREEN, RED, BLUE, YELLOW = (60, 180, 60), (40, 40, 220), (220, 120, 30), (0, 200, 230)


# ---------------------------------------------------------------- klasy i etykiety
def load_classes() -> list[str]:
    if not config.CLASSES_FILE.exists():
        raise FileNotFoundError(f"Brak pliku z klasami: {config.CLASSES_FILE}")
    names = [l.strip() for l in config.CLASSES_FILE.read_text(encoding="utf-8").splitlines()]
    return [n for n in names if n]


def list_images(folder: Path = config.IMAGES) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def read_yolo_labels(path: Path, width: int, height: int):
    """Etykiety YOLO (klasa xc yc w h, znormalizowane) -> ramki xyxy w pikselach."""
    boxes, classes = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, xc, yc, w, h = int(parts[0]), *map(float, parts[1:5])
        boxes.append([(xc - w / 2) * width, (yc - h / 2) * height,
                      (xc + w / 2) * width, (yc + h / 2) * height])
        classes.append(c)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(classes, dtype=int)


def write_yolo_labels(path: Path, boxes, classes, width: int, height: int):
    lines = []
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        lines.append(f"{int(c)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------------------------------------- IoU
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU = pole przecięcia / pole sumy, dla wszystkich par ramek xyxy (a: N×4, b: M×4)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


# --------------------------------------------------------------------- dopasowanie
def match_image(gt_boxes, gt_cls, pr_boxes, pr_scores, pr_cls, iou_thr=config.IOU_MATCH):
    """Dopasowuje wykrycia do wzorca w jednym obrazie.

    Zasady (jak na wykładzie): wykrycia idą od najwyższego wyniku; wykrycie jest TP, gdy
    ma IoU >= iou_thr z jeszcze nieprzypisanym wzorcem TEJ SAMEJ klasy. Jedna adnotacja
    daje najwyżej jedno TP. Reszta to FP, a niedopasowane adnotacje to FN.

    Zwraca słownik:
      tp   - lista (indeks wykrycia, indeks wzorca, IoU)
      fp   - lista (indeks wykrycia, typ błędu)
      fn   - lista indeksów wzorców
    Typy FP: 'tło', 'duplikat', 'zła klasa', 'słaba lokalizacja'.
    """
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    pr_boxes = np.asarray(pr_boxes, float).reshape(-1, 4)
    gt_cls, pr_cls = np.asarray(gt_cls, int), np.asarray(pr_cls, int)
    pr_scores = np.asarray(pr_scores, float)
    ious = iou_matrix(pr_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), bool)
    tp, fp = [], []
    for i in np.argsort(-pr_scores, kind="stable"):
        same = (gt_cls == pr_cls[i]) if len(gt_cls) else np.zeros(0, bool)
        free = same & ~used
        cand = np.where(free, ious[i], -1.0) if len(gt_boxes) else np.zeros(0)
        j = int(np.argmax(cand)) if len(cand) else -1
        if j >= 0 and cand[j] >= iou_thr:
            used[j] = True
            tp.append((int(i), j, float(cand[j])))
            continue
        # Dlaczego to FP?
        if len(gt_boxes):
            if (same & (ious[i] >= iou_thr)).any():
                kind = "duplikat"            # obiekt już wykryty wyżej
            elif ((~same) & (ious[i] >= iou_thr)).any():
                kind = "zła klasa"           # dobra ramka, ale inna klasa wzorca
            elif (same & (ious[i] > 0.1)).any():
                kind = "słaba lokalizacja"   # ta sama klasa, za małe pokrycie
            else:
                kind = "tło"
        else:
            kind = "tło"
        fp.append((int(i), kind))
    fn = [int(j) for j in np.where(~used)[0]]
    return {"tp": tp, "fp": fp, "fn": fn}


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


# ------------------------------------------------------------------------- model
def load_model(name: str):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(config.THREADS)
    return YOLO(name)


def class_map(model, class_names: list[str]) -> dict[int, int]:
    """ID klasy modelu (COCO) -> indeks w classes.txt. Pozostałe klasy modelu są ignorowane."""
    idx = {n: i for i, n in enumerate(class_names)}
    missing = [n for n in class_names if n not in model.names.values()]
    if missing:
        raise ValueError(f"Model nie zna klas: {missing}. Użyj nazw z COCO.")
    return {k: idx[v] for k, v in model.names.items() if v in idx}


def predict(model, img, imgsz, conf, cmap):
    """Jedna predykcja. Zwraca (ramki xyxy, wyniki, klasy w indeksach classes.txt, czasy)."""
    t0 = time.perf_counter()
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=config.NMS_IOU,
                      device=config.DEVICE, verbose=False)[0]
    wall_ms = (time.perf_counter() - t0) * 1000
    b = r.boxes
    xyxy = b.xyxy.cpu().numpy().reshape(-1, 4)
    sc = b.conf.cpu().numpy()
    cl = b.cls.cpu().numpy().astype(int)
    keep = np.array([c in cmap for c in cl], bool) if len(cl) else np.zeros(0, bool)
    xyxy, sc, cl = xyxy[keep], sc[keep], np.array([cmap[c] for c in cl[keep]], int)
    speed = dict(r.speed)  # preprocess / inference / postprocess [ms]
    speed["model_total"] = sum(r.speed.values())
    speed["wall"] = wall_ms
    return xyxy, sc, cl, speed


# ---------------------------------------------------------------------- rysowanie
def draw_boxes(img, boxes, labels, color=GREEN, thickness=2):
    for (x1, y1, x2, y2), text in zip(boxes, labels):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, thickness)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], y0 - th - 4), (p1[0] + tw + 4, y0), color, -1)
        cv2.putText(img, text, (p1[0] + 2, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img
