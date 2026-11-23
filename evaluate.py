"""Krok 3: główny eksperyment jakości.

Dla każdej pary (model, rozdzielczość) model liczy wykrycia na wszystkich oznaczonych
klatkach RAZ (przy niskim progu PREDICT_CONF). Potem dla każdego progu pewności z CONF_GRID
filtrujemy wykrycia i liczymy TP, FP, FN, precyzję, czułość, F1 oraz średnie IoU trafień.

Wyniki (katalog results/):
  metrics_by_conf.csv  - wszystkie konfiguracje i progi
  per_class.csv        - TP/FP/FN/precyzja/czułość dla każdej klasy przy REPORT_CONF
  timing.csv           - czas predykcji na obraz (średnia, mediana, 95. percentyl)
  predictions/*.json   - zapisane wykrycia (do visualize_errors.py)

Przykład:
    python evaluate.py
    python evaluate.py --models yolo11n.pt yolo26n.pt --imgsz 320 640
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd

import common
import config

FP_TYPES = ["tło", "duplikat", "zła klasa", "słaba lokalizacja"]


def load_dataset():
    """Wczytuje obrazy i wzorzec. Obrazy bez pliku .txt pomijamy (pusty plik = brak obiektów)."""
    data, skipped = [], 0
    for path in common.list_images():
        lab = config.LABELS / f"{path.stem}.txt"
        if not lab.exists():
            skipped += 1
            continue
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gb, gc = common.read_yolo_labels(lab, w, h)
        data.append({"path": path, "img": img, "gt_boxes": gb, "gt_cls": gc})
    if skipped:
        print(f"UWAGA: pominięto {skipped} obrazów bez etykiet.")
    if not data:
        raise SystemExit("Brak oznaczonych obrazów. Najpierw zrób kroki 1-2 (README).")
    return data


def evaluate_at_conf(data, preds, conf, n_cls):
    tp = np.zeros(n_cls, int); fp = np.zeros(n_cls, int); fn = np.zeros(n_cls, int)
    fp_types = {t: 0 for t in FP_TYPES}
    ious = []
    for d, p in zip(data, preds):
        keep = p["scores"] >= conf
        pb, ps, pc = p["boxes"][keep], p["scores"][keep], p["cls"][keep]
        m = common.match_image(d["gt_boxes"], d["gt_cls"], pb, ps, pc)
        for i, _, iou in m["tp"]:
            tp[pc[i]] += 1
            ious.append(iou)
        for i, kind in m["fp"]:
            fp[pc[i]] += 1
            fp_types[kind] += 1
        for j in m["fn"]:
            fn[d["gt_cls"][j]] += 1
    return tp, fp, fn, fp_types, ious


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--imgsz", nargs="+", type=int, default=config.IMGSZ)
    args = ap.parse_args()

    classes = common.load_classes()
    data = load_dataset()
    n_gt = sum(len(d["gt_cls"]) for d in data)
    print(f"Zbiór: {len(data)} obrazów, {n_gt} obiektów wzorcowych, {len(classes)} klas.")

    config.RESULTS.mkdir(exist_ok=True)
    (config.RESULTS / "predictions").mkdir(exist_ok=True)
    rows, class_rows, time_rows = [], [], []

    for mname in args.models:
        model = common.load_model(mname)
        cmap = common.class_map(model, classes)
        for imgsz in args.imgsz:
            tag = f"{mname.replace('.pt', '')}_{imgsz}"
            for _ in range(config.WARMUP):  # rozgrzewka (nie liczy się do czasu)
                common.predict(model, data[0]["img"], imgsz, config.PREDICT_CONF, cmap)
            preds, times = [], []
            for d in data:
                b, s, c, speed = common.predict(model, d["img"], imgsz, config.PREDICT_CONF, cmap)
                preds.append({"boxes": b, "scores": s, "cls": c})
                times.append(speed["wall"])
            times = np.array(times)
            time_rows.append({"model": mname, "imgsz": imgsz, "n_images": len(times),
                              "mean_ms": times.mean(), "median_ms": np.median(times),
                              "p95_ms": np.percentile(times, 95), "std_ms": times.std()})
            print(f"{tag}: {times.mean():.1f} ms/obraz (p95 {np.percentile(times, 95):.1f} ms)")

            with open(config.RESULTS / "predictions" / f"{tag}.json", "w") as f:
                json.dump([{"path": d["path"].name, "boxes": p["boxes"].tolist(),
                            "scores": p["scores"].tolist(), "cls": p["cls"].tolist()}
                           for d, p in zip(data, preds)], f)

            for conf in config.CONF_GRID:
                tp, fp, fn, fpt, ious = evaluate_at_conf(data, preds, conf, len(classes))
                P, R, F1 = common.prf(tp.sum(), fp.sum(), fn.sum())
                rows.append({"model": mname, "imgsz": imgsz, "conf": conf,
                             "tp": tp.sum(), "fp": fp.sum(), "fn": fn.sum(),
                             "precision": P, "recall": R, "f1": F1,
                             "mean_iou_tp": float(np.mean(ious)) if ious else 0.0,
                             **{f"fp_{k}": v for k, v in fpt.items()}})
                if abs(conf - config.REPORT_CONF) < 1e-9:
                    for ci, cname in enumerate(classes):
                        p, r, f1 = common.prf(tp[ci], fp[ci], fn[ci])
                        class_rows.append({"model": mname, "imgsz": imgsz, "class": cname,
                                           "n_gt": int(tp[ci] + fn[ci]), "tp": int(tp[ci]),
                                           "fp": int(fp[ci]), "fn": int(fn[ci]),
                                           "precision": p, "recall": r, "f1": f1})

    pd.DataFrame(rows).to_csv(config.RESULTS / "metrics_by_conf.csv", index=False)
    pd.DataFrame(class_rows).to_csv(config.RESULTS / "per_class.csv", index=False)
    pd.DataFrame(time_rows).to_csv(config.RESULTS / "timing.csv", index=False)

    df = pd.DataFrame(rows)
    show = df[np.isclose(df["conf"], config.REPORT_CONF)]
    print(f"\n=== Wyniki przy progu {config.REPORT_CONF} (IoU >= {config.IOU_MATCH}) ===")
    print(show[["model", "imgsz", "tp", "fp", "fn", "precision", "recall", "f1", "mean_iou_tp"]]
          .round(3).to_string(index=False))
    print(f"\nZapisano CSV w {config.RESULTS}. Dalej: benchmark_fps.py, make_plots.py, visualize_errors.py")


if __name__ == "__main__":
    main()
