"""Krok 7 (bonus): film z narysowanymi wykryciami.

Przykład:
    python -m visualization.render_video --model yolo11n.pt --imgsz 640 --conf 0.5 --out results/videos/biurko_wynik.mp4
"""
import argparse
import subprocess

import cv2

import common
import config

import cv2

# Wyraźniejsze kolory BGR (OpenCV używa BGR, nie RGB)
CLASS_COLORS = {
    "laptop": (255, 140, 0),       # pomarańczowy
    "cell phone": (255, 0, 255),   # magenta
    "keyboard": (0, 200, 255),     # żółto-niebieski / amber
    "mouse": (255, 80, 80),        # jasny czerwony
    "book": (180, 0, 255),         # fioletowy
    "cup": (0, 180, 0),            # ciemniejszy zielony
    "bottle": (255, 255, 0),       # cyjan
    "chair": (0, 120, 255),        # pomarańczowo-niebieski
    "default": (0, 255, 255),      # żółty
}


def draw_boxes_colored(img, boxes, scores, classes_idx, class_names):
    """
    Rysuje grubsze ramki i większe etykiety w różnych kolorach.
    """
    for (x1, y1, x2, y2), score, cls_idx in zip(boxes, scores, classes_idx):
        cls_name = class_names[cls_idx]
        color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # grubsza ramka
        thickness = 4
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # większy napis
        label = f"{cls_name} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # tło pod napisem
        text_y = max(y1, th + 10)
        cv2.rectangle(
            img,
            (x1, text_y - th - 10),
            (x1 + tw + 10, text_y),
            color,
            -1
        )

        # biały napis
        cv2.putText(
            img,
            label,
            (x1 + 5, text_y - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return img

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", default=str(config.VIDEO))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default=str(config.VIDEOS / "biurko_dokladniejszy_0-35.mp4"))
    args = ap.parse_args()

    classes = common.load_classes()
    model = common.load_model(args.model)
    cmap = common.class_map(model, classes)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Nie można otworzyć filmu: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    config.VIDEOS.mkdir(parents=True, exist_ok=True)
    config.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw = config.VIDEOS / "_raw.avi"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b, s, c, sp = common.predict(model, frame, args.imgsz, args.conf, cmap)
        common.draw_boxes_colored(frame, b, s, c, classes)
        cv2.putText(
            frame,
            f"{args.model}  {args.imgsz}px  conf>={args.conf}  {sp['wall']:.0f} ms",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )
        writer.write(frame)
        n += 1
    cap.release(); writer.release()

    # Konwersja do MP4/H.264 (odtwarza się w przeglądarce), jeśli dostępny jest imageio-ffmpeg.
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        subprocess.run([get_ffmpeg_exe(), "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-an", "-movflags", "+faststart", args.out], check=True, capture_output=True)
        raw.unlink()
        print(f"Zapisano {n} klatek: {args.out}")
    except Exception as e:  # noqa: BLE001
        print(f"Zapisano {n} klatek jako {raw} (konwersja do MP4 nieudana: {e})")


if __name__ == "__main__":
    main()
