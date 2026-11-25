"""FPS na pierwszym planie, F1 w tle dla 12 konfiguracji.

Uruchom z katalogu projektu: python -m visualization.make_fps_f1_bars
Każda para model–imgsz korzysta z progu najlepszego F1.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import config


def main():
    config.ensure_result_dirs()
    with (config.TABLES / "metrics_by_conf.csv").open(encoding="utf-8") as stream:
        quality = list(csv.DictReader(stream))
    benchmark_path = config.latest_benchmark_file()
    if not benchmark_path.exists():
        raise ValueError("Brak benchmarku FPS")
    with benchmark_path.open(encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    speeds = {(r["model"], int(r["imgsz"]), round(float(r["conf"]), 8)): r for r in records}
    if len(speeds) != len(records):
        raise ValueError("Powtórzone konfiguracje benchmarku")
    selected = []
    for model in config.MODELS:
        for size in config.IMGSZ:
            best = max((r for r in quality if r["model"] == model and int(r["imgsz"]) == size),
                       key=lambda r: float(r["f1"]))
            conf = float(best["conf"])
            speed = speeds[model, size, round(conf, 8)]
            selected.append(dict(model=model, imgsz=size, conf=conf,
                                 f1_percent=float(best["f1"]) * 100,
                                 fps_pipeline=float(speed["fps_pipeline"])))

    blue, orange = "#245B86", "#C77B16"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, ax_fps = plt.subplots(figsize=(15, 7.8))
        ax_f1 = ax_fps.twinx()
        # Tło jakości pod osią i słupkami szybkości.
        ax_fps.set_zorder(2)
        ax_fps.patch.set_visible(False)
        ax_f1.set_zorder(1)
        x = list(range(len(selected)))
        f1_values = [r["f1_percent"] for r in selected]
        fps_values = [r["fps_pipeline"] for r in selected]
        quality_bars = ax_f1.bar(x, f1_values, width=.78, color=orange, alpha=.22,
                                 edgecolor=orange, linewidth=.8)
        speed_bars = ax_fps.bar(x, fps_values, width=.37, color=blue)
        ax_fps.set_ylim(0, 35)
        ax_f1.set_ylim(0, 110)
        ax_f1.set_yticks([0, 20, 40, 60, 80, 100])
        ax_fps.set_ylabel("FPS całego potoku", color=blue, fontsize=13, labelpad=12)
        ax_f1.set_ylabel("F1 [%]", color=orange, fontsize=13, labelpad=12)
        ax_fps.tick_params(axis="y", colors=blue)
        ax_f1.tick_params(axis="y", colors=orange)
        ax_fps.axhline(config.REALTIME_FPS, color="#B34040", linestyle="--", linewidth=1.4)
        ax_fps.bar_label(speed_bars,
                         labels=[f"{v:.2f}".replace(".", ",") for v in fps_values],
                         label_type="edge", padding=4, fontsize=10, color=blue, fontweight="bold")
        for index, (f1, fps) in enumerate(zip(f1_values, fps_values)):
            obscured = fps / 35 >= f1 / 110
            ax_f1.annotate(f"{f1:.1f}%".replace(".", ","),
                           (index - .24 if obscured else index, f1),
                           xytext=(0, 5), textcoords="offset points",
                           ha="right" if obscured else "center", va="bottom",
                           fontsize=10, color="#8D540B")
        labels = [f"{r['model'][:-3]}\n{r['imgsz']} px\nconf {r['conf']:.2f}".replace(".", ",")
                  for r in selected]
        ax_fps.set_xticks(x, labels, fontsize=10)
        ax_fps.tick_params(axis="x", length=0, pad=12)
        ax_fps.set_xlim(-.7, len(selected) - .3)
        ax_fps.grid(axis="y", alpha=.16)
        ax_fps.set_axisbelow(True)
        ax_fps.spines[["top", "right"]].set_visible(False)
        ax_f1.spines[["top", "left"]].set_visible(False)
        for boundary in [2.5, 5.5, 8.5]:
            ax_fps.axvline(boundary, color="#999999", linewidth=.7, alpha=.3)
        fig.suptitle("Jakość detekcji a szybkość przetwarzania", fontsize=20, y=.98, va="top")
        fig.text(.5, .905, "12 konfiguracji · dla każdej pary model–rozdzielczość próg o najwyższym F1",
                 ha="center", va="top", fontsize=12)
        fig.legend(handles=[Patch(facecolor=blue, label="FPS — słupek na pierwszym planie, lewa oś"),
                            Patch(facecolor=orange, alpha=.22, label="F1 — słupek w tle, prawa oś"),
                            Line2D([0], [0], color="#B34040", linestyle="--", label="Wymaganie 25 FPS")],
                   loc="upper center", bbox_to_anchor=(.5, .85), ncol=3, frameon=False, fontsize=10)
        fig.text(.5, .035,
                 "FPS: odczyt + predykcja + rysowanie, 300 klatek · F1: 31 oznaczonych obrazów, IoU ≥ 0,50",
                 ha="center", fontsize=10, color="#444444")
        fig.text(.5, .011,
                 "Osie mają różne jednostki i skale; wysokości FPS i F1 nie są bezpośrednio porównywalne.",
                 ha="center", fontsize=10, color="#444444")
        fig.subplots_adjust(left=.07, right=.935, top=.76, bottom=.19)
        output = config.PLOTS / "fps_vs_f1_bars.png"
        fig.savefig(output, dpi=220, facecolor="white")
        plt.close(fig)
    with (config.TABLES / "fps_vs_f1_bars.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(output)


if __name__ == "__main__":
    main()
