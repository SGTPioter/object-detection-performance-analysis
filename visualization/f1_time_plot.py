"""Czytelny wykres F1–czas dla siatki konfiguracji projektu."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_f1_vs_time(records, output):
    # Rozmieszczenie podpisów jest stałe dla badanej siatki, w jednostkach osi.
    positions = {
        ('yolo11n.pt', 320): (19, 60.4),
        ('yolo26n.pt', 320): (28, 69.0),
        ('yolo11n.pt', 640): (23, 72.4),
        ('yolo26n.pt', 640): (48, 74.5),
        ('yolo26n.pt', 960): (41, 79.0),
        ('yolo11n.pt', 960): (72, 80.4),
        ('yolo11s.pt', 320): (25, 80.8),
        ('yolo26s.pt', 320): (25, 86.7),
        ('yolo11s.pt', 640): (78, 84.9),
        ('yolo26s.pt', 640): (53, 91.4),
        ('yolo11s.pt', 960): (109, 85.0),
        ('yolo26s.pt', 960): (139, 91.4),
    }
    colors = {'yolo11n.pt': '#52728F', 'yolo11s.pt': '#95634C',
              'yolo26n.pt': '#8A659C', 'yolo26s.pt': '#007F82'}
    markers = {320: 'o', 640: 's', 960: 'D'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
        fig, ax = plt.subplots(figsize=(10, 6.8))
        ax.set(xlim=(5, 165), ylim=(58, 94),
               xlabel='Średni czas predykcji na obraz [ms]',
               ylabel='F1 przy najlepszym badanym progu [%]')
        ax.set_title('Kompromis jakości i czasu', fontsize=17, pad=18)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        for r in records:
            model, size = r['model'], int(r['imgsz'])
            x, y = float(r['mean_ms']), float(r['f1']) * 100
            color = colors[model]
            ax.scatter(x, y, s=85, color=color, marker=markers[size],
                       edgecolors='white', linewidths=.8, zorder=4)
            conf = f"{float(r['conf']):.2f}".replace('.', ',')
            label = f"{model[:-3]} · {size}\nconf {conf}"
            ax.annotate(label, (x, y), xytext=positions[(model, size)],
                        textcoords='data', ha='center', va='center',
                        fontsize=10, color='#20272C', linespacing=1.25,
                        bbox=dict(boxstyle='round,pad=.28', fc='white', ec='none', alpha=.96),
                        arrowprops=dict(arrowstyle='-', color=color, lw=1,
                                        shrinkA=4, shrinkB=7), zorder=5)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(pad=1.6)
        fig.savefig(output, dpi=220, facecolor='white')
        plt.close(fig)
