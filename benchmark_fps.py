"""Krok 4: pomiar FPS na całym nagraniu (cały potok: odczyt + model + rysowanie).

Dla każdej pary (model, rozdzielczość) przetwarzamy kolejne klatki filmu i mierzymy:
  read_ms    - odczyt klatki z pliku
  predict_ms - przygotowanie obrazu + model + filtrowanie (NMS)
  draw_ms    - rysowanie ramek
  FPS        - liczba klatek / łączny czas całego potoku (nie tylko samego modelu!)

Wynik: results/benchmark_fps.csv

Przykład:
    python benchmark_fps.py --max-frames 300
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd

import common
import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", default=str(config.VIDEO))
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--imgsz", nargs="+", type=int, default=config.IMGSZ)
    ap.add_argument("--conf", type=float, default=config.REPORT_CONF)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    classes = common.load_classes()
    rows = []
    for mname in args.models:
        model = common.load_model(mname)
        cmap = common.class_map(model, classes)
        for imgsz in args.imgsz:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SystemExit(f"Nie można otworzyć filmu: {args.video}")
            ok, first = cap.read()
            for _ in range(config.WARMUP):  # rozgrzewka
                common.predict(model, first, imgsz, args.conf, cmap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            t_read, t_pred, t_draw, n = [], [], [], 0
            t_start = time.perf_counter()
            while n < args.max_frames:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                t1 = time.perf_counter()
                b, s, c, _ = common.predict(model, frame, imgsz, args.conf, cmap)
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
            rows.append({"model": mname, "imgsz": imgsz, "frames": n,
                         "read_ms": np.mean(t_read) * 1000, "predict_ms": pred_ms.mean(),
                         "predict_p95_ms": np.percentile(pred_ms, 95),
                         "draw_ms": np.mean(t_draw) * 1000,
                         "fps_pipeline": n / total, "fps_model_only": 1000 / pred_ms.mean()})
            r = rows[-1]
            print(f"{mname} @ {imgsz}: pipeline {r['fps_pipeline']:.1f} FPS | "
                  f"sam model {r['fps_model_only']:.1f} FPS | predict {r['predict_ms']:.1f} ms")

    config.RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "benchmark_fps.csv", index=False)
    print(f"\nZapisano {config.RESULTS / 'benchmark_fps.csv'}")


if __name__ == "__main__":
    main()
