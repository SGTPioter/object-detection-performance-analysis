"""Krok 6: wykresy i podsumowanie (results/plots/*.png oraz results/summary.md).

Wykresy:
  f1_vs_time.png        - kompromis jakość/czas (każdy punkt = model + rozdzielczość, najlepszy próg)
  precision_recall.png  - precyzja i czułość w funkcji progu pewności
  pr_curve.png          - precyzja vs czułość dla kolejnych progów
  fps.png               - FPS całego potoku z benchmark_fps.py (jeśli plik istnieje)

W summary.md wskazujemy:
  * konfigurację "czasu rzeczywistego": najwyższe F1 przy FPS >= REALTIME_FPS,
  * konfigurację najwyższej jakości: najwyższe F1 w całym porównaniu.

Przykład:
    python -m visualization.make_plots
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from visualization.f1_time_plot import plot_f1_vs_time

R = config.RESULTS


def label(m, s):
    return f"{m.replace('.pt', '')} · {s}"


def main():
    config.ensure_result_dirs()
    met = pd.read_csv(config.TABLES / "metrics_by_conf.csv")
    tim = pd.read_csv(config.BENCHMARKS / "timing.csv")
    fps_file = config.latest_benchmark_file()
    if fps_file.exists():
        print(f"Benchmark FPS: {fps_file}")
    fps = pd.read_csv(fps_file) if fps_file.exists() else None

    # Najlepszy próg dla każdej pary (model, imgsz) wg F1.
    best = met.loc[met.groupby(["model", "imgsz"])["f1"].idxmax()].merge(tim, on=["model", "imgsz"])
    if fps is not None:
        if "conf" not in fps.columns:
            raise ValueError("Benchmark nie zawiera conf. Uruchom ponownie benchmark_fps.py.")
        best = best.merge(
            fps[["model", "imgsz", "conf", "fps_pipeline", "fps_model_only"]],
            on=["model", "imgsz", "conf"], how="left", validate="one_to_one",
        )
        if best[["fps_pipeline", "fps_model_only"]].isna().any().any():
            raise ValueError(
                "Brakuje FPS dla konfiguracji o najlepszym F1. "
                "Uruchom benchmark_fps.py bez --conf dla wszystkich modeli i rozdzielczości."
            )
    best["name"] = [label(m, s) for m, s in zip(best["model"], best["imgsz"])]

    # 1) F1 vs czas
    plot_f1_vs_time(best.to_dict('records'), config.PLOTS / 'f1_vs_time.png')

    # 2) precyzja i czułość vs próg
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for (m, s), g in met.groupby(["model", "imgsz"]):
        for a, col, name in zip(ax, ["precision", "recall", "f1"], ["Precyzja", "Czułość", "F1"]):
            a.plot(g["conf"], g[col] * 100, marker="o", label=label(m, s), linestyle="-" if s == max(config.IMGSZ) else "--")
            a.set_title(name); a.set_xlabel("Próg pewności"); a.set_ylabel("%"); a.grid(alpha=.3)
    ax[0].legend(fontsize=7); plt.tight_layout()
    plt.savefig(config.PLOTS / "precision_recall.png", dpi=150); plt.close()

    # 3) krzywa PR
    plt.figure(figsize=(6.5, 5.5))
    for (m, s), g in met.groupby(["model", "imgsz"]):
        plt.plot(g["recall"] * 100, g["precision"] * 100, marker="o", label=label(m, s), linestyle="-" if s == max(config.IMGSZ) else "--")
    plt.xlabel("Czułość [%]"); plt.ylabel("Precyzja [%]"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.title("Precyzja vs czułość (kolejne progi)"); plt.tight_layout()
    plt.savefig(config.PLOTS / "pr_curve.png", dpi=150); plt.close()

    # 4) FPS
    if fps is not None:
        plt.figure(figsize=(8, 4.5))
        names = [label(m, s) for m, s in zip(fps["model"], fps["imgsz"])]
        plt.bar(names, fps["fps_pipeline"]); plt.axhline(config.REALTIME_FPS, color="red", ls="--", label=f"{config.REALTIME_FPS} FPS")
        plt.ylabel("FPS całego potoku"); plt.xticks(rotation=30, ha="right"); plt.legend(); plt.tight_layout()
        plt.savefig(config.PLOTS / "fps.png", dpi=150); plt.close()

    # Podsumowanie i wybór konfiguracji.
    lines = ["# Podsumowanie eksperymentu", "",
             f"Dopasowanie: IoU >= {config.IOU_MATCH}, jedna adnotacja = najwyżej jedno TP. "
             "Dla każdej konfiguracji pokazano próg pewności o najwyższym F1.", ""]
    cols = ["name", "conf", "precision", "recall", "f1", "mean_iou_tp", "fp", "fn", "mean_ms", "p95_ms"]
    if fps is not None:
        cols += ["fps_pipeline"]
    tab = best[cols].copy()
    for c in ["precision", "recall", "f1"]:
        tab[c] = (tab[c] * 100).round(1)
    lines.append(tab.round(3).to_markdown(index=False) if _has_tabulate() else tab.round(3).to_string(index=False))
    lines.append("")

    best_q = best.loc[best["f1"].idxmax()]
    lines.append(f"**Najwyższa jakość:** {best_q['name']}, conf {best_q['conf']:.2f} "
                 f"(F1 {best_q['f1']*100:.1f}%, precyzja {best_q['precision']*100:.1f}%, "
                 f"czułość {best_q['recall']*100:.1f}%, {best_q['mean_ms']:.1f} ms/obraz).")
    if fps is not None and "fps_pipeline" in best:
        rt = best[best["fps_pipeline"] >= config.REALTIME_FPS]
        if len(rt):
            r = rt.loc[rt["f1"].idxmax()]
            lines.append(f"**Czas rzeczywisty (>= {config.REALTIME_FPS} FPS):** {r['name']}, conf {r['conf']:.2f} "
                         f"(F1 {r['f1']*100:.1f}%, {r['fps_pipeline']:.1f} FPS całego potoku).")
        else:
            r = best.loc[best["fps_pipeline"].idxmax()]
            lines.append(f"Żadna konfiguracja nie osiągnęła {config.REALTIME_FPS} FPS. Najszybsza: {r['name']} "
                         f"({r['fps_pipeline']:.1f} FPS, F1 {r['f1']*100:.1f}%).")
    else:
        lines.append("(Uruchom benchmark_fps.py, aby wskazać konfigurację czasu rzeczywistego.)")
    lines += ["", "> Próg pewności wybrano na tych samych danych, na których liczono metryki, "
              "więc wynik jest nieco optymistyczny. Zaznacz to w raporcie."]
    (R / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nZapisano wykresy i {R / 'summary.md'}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
