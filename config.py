
"""Wspólna konfiguracja projektu: ścieżki, modele i parametry eksperymentu."""

import os
from pathlib import Path

# --- Ścieżki ---------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA = Path(os.environ.get("P3_DATA", ROOT / "data"))

VIDEO = DATA / "video" / "biurko.mp4"
IMAGES = DATA / "images"
LABELS = DATA / "labels"
CLASSES_FILE = DATA / "classes.txt"

RESULTS = Path(os.environ.get("P3_RESULTS", ROOT / "results"))


# --- Modele ----------------------------------------------------

MODELS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo26n.pt",
    "yolo26s.pt"
]


# --- Parametry eksperymentu ------------------------------------

# Rozdzielczości wejściowe
IMGSZ = [320, 640, 960]

# Progi pewności
CONF_GRID = [
    0.10, 0.20, 0.30, 0.35, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90
]

# Minimalny próg używany do zbierania predykcji
PREDICT_CONF = 0.05

# Próg do szczegółowych tabel i domyślnego benchmarku FPS
REPORT_CONF = 0.50

# Próg NMS
NMS_IOU = 0.70

# Minimalne IoU dla poprawnego wykrycia (TP)
IOU_MATCH = 0.50


# --- Sprzęt i pomiary ------------------------------------------

DEVICE = "cpu"
THREADS = 4
WARMUP = 5

# Minimalny FPS wymagany dla działania w czasie rzeczywistym
REALTIME_FPS = 25
