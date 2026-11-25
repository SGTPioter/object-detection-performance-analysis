"""Krok 4: pomiar FPS na całym nagraniu (cały potok: odczyt + model + rysowanie).

Dla każdej pary (model, rozdzielczość) przetwarzamy kolejne klatki filmu i mierzymy:
  read_ms    - odczyt klatki z pliku
  predict_ms - przygotowanie obrazu + model + filtrowanie (NMS)
  draw_ms    - rysowanie ramek
  FPS        - liczba klatek / łączny czas całego potoku (nie tylko samego modelu!)

Wynik: results/benchmarks/benchmark_fps.csv
Bez --conf próg jest wybierany według najlepszego F1 w metrics_by_conf.csv.

Przykład:
    python -m experiments.benchmark_fps --max-frames 300
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import common
import config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", default=str(config.VIDEO))
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--imgsz", nargs="+", type=int, default=config.IMGSZ)
    ap.add_argument("--conf", type=float, default=None,
                    help="Próg pewności; domyślnie próg najlepszego F1 dla modelu i imgsz")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--output", type=Path, default=config.BENCHMARKS / "benchmark_fps.csv",
                    help="Plik wynikowy CSV (osobna ścieżka dla pomiarów dodatkowych)")
    args = ap.parse_args(argv)
    if args.max_frames <= 0:
        ap.error("--max-frames musi być większe od zera")
    if args.conf is not None and not 0 <= args.conf <= 1:
        ap.error("--conf musi należeć do przedziału [0, 1]")

    metrics_file = config.TABLES / "metrics_by_conf.csv"
    if args.conf is None and not metrics_file.exists():
        ap.error("Brak metrics_by_conf.csv. Uruchom evaluate.py lub podaj --conf.")
    metrics = pd.read_csv(metrics_file) if args.conf is None else None

    # Sprawdź komplet konfiguracji przed rozpoczęciem kosztownych pomiarów.
    thresholds = {}
    for mname in args.models:
        for imgsz in args.imgsz:
            if args.conf is not None:
                conf = args.conf
            else:
                candidates = metrics[(metrics["model"] == mname) & (metrics["imgsz"] == imgsz)]
                if candidates.empty:
                    ap.error(f"Brak metryk jakości dla {mname}, imgsz={imgsz}")
                conf = float(candidates.loc[candidates["f1"].idxmax(), "conf"])
            thresholds[mname, imgsz] = conf

    classes = common.load_classes()
    rows = []
    for mname in args.models:
        weights = config.ROOT / "models" / mname
        model = common.load_model(str(weights) if weights.is_file() else mname)
        cmap = common.class_map(model, classes)
        for imgsz in args.imgsz:
            conf = thresholds[mname, imgsz]
            print(f"Benchmark: {mname}, imgsz={imgsz}, conf={conf:.2f}")
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SystemExit(f"Nie można otworzyć filmu: {args.video}")
            ok, first = cap.read()
            if not ok:
                cap.release()
                raise SystemExit(f"Nie można odczytać pierwszej klatki: {args.video}")
            for _ in range(config.WARMUP):  # rozgrzewka
                common.predict(model, first, imgsz, conf, cmap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            t_read, t_pred, t_draw, n = [], [], [], 0
            t_start = time.perf_counter()
            while n < args.max_frames:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                t1 = time.perf_counter()
                b, s, c, _ = common.predict(model, frame, imgsz, conf, cmap)
                t2 = time.perf_counter()
                common.draw_boxes(frame, b, [f"{classes[k]} {v:.2f}" for k, v in zip(c, s)])
                t3 = time.perf_counter()
                t_read.append(t1 - t0); t_pred.append(t2 - t1); t_draw.append(t3 - t2)
                n += 1
            total = time.perf_counter() - t_start
            cap.release()
            if n == 0:
                continue
            pred_ms = np.array(t_pred) * 1000
            rows.append({"model": mname, "imgsz": imgsz, "conf": conf, "frames": n,
                         "read_ms": np.mean(t_read) * 1000, "predict_ms": pred_ms.mean(),
                         "predict_p95_ms": np.percentile(pred_ms, 95),
                         "draw_ms": np.mean(t_draw) * 1000,
                         "fps_pipeline": n / total, "fps_model_only": 1000 / pred_ms.mean()})
            r = rows[-1]
            print(f"{mname} @ {imgsz}, conf {conf:.2f}: pipeline {r['fps_pipeline']:.1f} FPS | "
                  f"sam model {r['fps_model_only']:.1f} FPS | predict {r['predict_ms']:.1f} ms")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Najpierw zachowaj pomiary pod unikalną nazwą. Blokada głównego CSV
    # nie może spowodować utraty całego zakończonego benchmarku.
    archive_dir = output.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{output.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}{output.suffix}"
    df.to_csv(archive, index=False)
    print(f"\nZapisano kopię wyników: {archive}")
    try:
        df.to_csv(output, index=False)
    except PermissionError:
        print(f"Nie można nadpisać {output} (blokada lub tylko do odczytu). "
              f"Wyniki są zachowane w {archive}.")
    else:
        print(f"Zapisano {output}")


if __name__ == "__main__":
    main()
