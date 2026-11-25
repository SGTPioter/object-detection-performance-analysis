"""Średnie FN, FP i F1 z rozdzielczości, przy progach najlepszego F1.

python -m visualization.make_class_average
Korzysta z zapisanych predykcji, bez ponownego uruchamiania modeli.
"""
import argparse
import csv
import json

import config

MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt"]
CLASSES = ["laptop", "cell phone", "keyboard", "mouse", "book", "cup"]
NAMES = ["Laptop", "Telefon", "Klawiatura", "Mysz", "Książka", "Kubek"]
COLORS = ["#4477AA", "#EE7733", "#AA3377", "#228833"]
DETAIL = config.TABLES / "per_class_best_conf.csv"
AVERAGE = config.TABLES / "per_class_average.csv"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calculate():
    import cv2
    import numpy as np
    import common
    from experiments import evaluate

    classes = common.load_classes()
    assert classes == CLASSES, "Inna kolejność klas niż oczekiwana"
    data = {}
    for path in common.list_images():
        label = config.LABELS / f"{path.stem}.txt"
        if not label.exists():
            continue
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        boxes, cls = common.read_yolo_labels(label, width, height)
        data[path.name] = {"gt_boxes": boxes, "gt_cls": cls}
    with (config.TABLES / "metrics_by_conf.csv").open(encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    detailed = []
    for model in MODELS:
        for size in config.IMGSZ:
            best = max((r for r in metrics if r["model"] == model and int(r["imgsz"]) == size),
                       key=lambda r: float(r["f1"]))
            conf = float(best["conf"])
            path = config.PREDICTIONS / f"{model[:-3]}_{size}.json"
            predictions = json.loads(path.read_text(encoding="utf-8"))
            assert len(predictions) == len(data)
            assert {r["path"] for r in predictions} == set(data)
            ordered = [data[r["path"]] for r in predictions]
            parsed = [{"boxes": np.asarray(r["boxes"], float).reshape(-1, 4),
                       "scores": np.asarray(r["scores"], float),
                       "cls": np.asarray(r["cls"], int)} for r in predictions]
            tp, fp, fn, _, _ = evaluate.evaluate_at_conf(ordered, parsed, conf, len(classes))
            for key, values in [("tp", tp), ("fp", fp), ("fn", fn)]:
                assert int(values.sum()) == int(best[key]), (model, size, key)
            for i, cls in enumerate(classes):
                _, _, f1 = common.prf(tp[i], fp[i], fn[i])
                detailed.append(dict(model=model, imgsz=size, conf=conf, **{"class": cls},
                                     tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]), f1=float(f1)))
    averaged = []
    for model in MODELS:
        for cls in classes:
            group = [r for r in detailed if r["model"] == model and r["class"] == cls]
            averaged.append(dict(model=model, **{"class": cls}, n_configurations=len(group),
                                 mean_fn=sum(r["fn"] for r in group) / len(group),
                                 mean_fp=sum(r["fp"] for r in group) / len(group),
                                 mean_f1=sum(r["f1"] for r in group) / len(group)))
    write_csv(DETAIL, detailed)
    write_csv(AVERAGE, averaged)
    print("Sprawdzono zgodność TP/FP/FN wszystkich 12 konfiguracji z metrics_by_conf.csv.")


def plot():
    config.PLOTS.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    with AVERAGE.open(encoding="utf-8") as stream:
        lookup = {(r["model"], r["class"]): r for r in csv.DictReader(stream)}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10.5))
        fig.suptitle("Średni bilans modeli według klas", fontsize=20, y=.985, va="top")
        fig.text(.5, .925, "Średnia z imgsz 320, 640 i 960 · dla każdej konfiguracji próg najlepszego F1",
                 ha="center", va="top", fontsize=12)
        x, width = np.arange(6), .19
        for ax, metric, title in zip(axes, ["mean_fn", "mean_fp", "mean_f1"],
                                     ["Średnie FN · pominięte obiekty — mniej = lepiej",
                                      "Średnie FP · fałszywe wykrycia — mniej = lepiej",
                                      "Średnie F1 · jakość detekcji — więcej = lepiej"]):
            highest = 0
            for i, (model, color) in enumerate(zip(MODELS, COLORS)):
                values = [float(lookup[model, cls][metric]) * (100 if metric == "mean_f1" else 1)
                          for cls in CLASSES]
                highest = max(highest, max(values))
                bars = ax.bar(x + (i - 1.5) * width, values, width=width,
                              label=model[:-3].upper(), color=color, zorder=3)
                ax.bar_label(bars, labels=[f"{v:.1f}".replace(".", ",") for v in values],
                             padding=3, fontsize=8)
            ax.set_title(title, loc="left", fontsize=13, pad=12)
            ax.set_ylabel("Średnie F1 [%]" if metric == "mean_f1" else "Średnia liczba")
            ax.set_ylim(0, 112 if metric == "mean_f1" else max(1, highest) * 1.22)
            ax.set_xticks(x, NAMES)
            ax.tick_params(axis="x", length=0, pad=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2, zorder=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .885),
                   ncol=4, frameon=False, fontsize=12)
        fig.text(.5, .016,
                 "Równe wagi trzech rozdzielczości · F1 to średnia wyników F1, a nie F1 ze zsumowanych zliczeń · IoU ≥ 0,50",
                 ha="center", fontsize=10, color="#444444")
        fig.subplots_adjust(top=.80, bottom=.07, left=.07, right=.985, hspace=.56)
        output = config.PLOTS / "per_class_average_fn_fp_f1.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        calculate()
    if not args.data_only:
        plot()
