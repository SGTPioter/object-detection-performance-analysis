"""Krok 7 (bonus): film z narysowanymi wykryciami.

Przykład:
    python render_video.py --model yolo11n.pt --imgsz 640 --conf 0.5 --out results/biurko_wynik.mp4
"""
import argparse
import subprocess

import cv2

import common
import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", default=str(config.VIDEO))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default=str(config.RESULTS / "biurko_dokladniejszy_0-35.mp4"))
    args = ap.parse_args()

    classes = common.load_classes()
    model = common.load_model(args.model)
    cmap = common.class_map(model, classes)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Nie można otworzyć filmu: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    config.RESULTS.mkdir(exist_ok=True)
    raw = config.RESULTS / "_raw.avi"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b, s, c, sp = common.predict(model, frame, args.imgsz, args.conf, cmap)
        common.draw_boxes(frame, b, [f"{classes[k]} {v:.2f}" for k, v in zip(c, s)])
        cv2.putText(frame, f"{args.model} {args.imgsz}px conf>={args.conf}  {sp['wall']:.0f} ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
