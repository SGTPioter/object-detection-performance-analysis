
import subprocess
import cv2

import common
import config

# Konfiguracje wybrane na podstawie wyników eksperymentu
CONFIGURATIONS = [
    ("yolo26s.pt", 960, 0.35),  # Najwyższe F1
    ("yolo26s.pt", 960, 0.10),  # Najwyższy recall
    ("yolo26s.pt", 640, 0.30),  # Kompromis jakość/szybkość
    ("yolo26n.pt", 320, 0.10),  # Czas rzeczywisty
]

# Ile sekund oryginalnego filmu przetworzyć
MAX_SECONDS = 60

# Katalog wynikowy
OUTPUT_DIR = config.VIDEOS


def render_video(model, model_name, imgsz, conf, classes, cmap):
    cap = cv2.VideoCapture(str(config.VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Nie można otworzyć filmu.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Film wynikowy w Full HD dla szybszego zapisu
    if width >= height:
        output_size = (1920, 1080)
    else:
        output_size = (1080, 1920)

    name = (
        f"{model_name.replace('.pt', '')}_"
        f"{imgsz}_conf{round(conf * 100):03d}"
    )

    raw_path = OUTPUT_DIR / f"{name}_raw.avi"
    output_path = OUTPUT_DIR / f"{name}.mp4"

    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        output_size
    )

    total_frames = min(
        int(MAX_SECONDS * fps),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )

    print(f"\nRenderowanie: {name}")

    for i in range(total_frames):
        success, frame = cap.read()

        if not success:
            break

        boxes, scores, detected_classes, _ = common.predict(
            model, frame, imgsz, conf, cmap
        )

        labels = [
            f"{classes[c]} {score:.2f}"
            for c, score in zip(detected_classes, scores)
        ]

        common.draw_boxes_colored(
            frame,
            boxes,
            scores,
            detected_classes,
            classes
        )

        frame = cv2.resize(frame, output_size)

        cv2.putText(
            frame,
            f"{model_name} | imgsz={imgsz} | conf={conf}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total_frames} klatek")

    cap.release()
    writer.release()

    # Konwersja AVI do MP4
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        subprocess.run(
            [
                get_ffmpeg_exe(),
                "-y",
                "-i", str(raw_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an",
                "-movflags", "+faststart",
                str(output_path)
            ],
            check=True,
            capture_output=True
        )

        raw_path.unlink()
        print(f"Zapisano: {output_path}")

    except Exception as e:
        print(f"Nie udało się przekonwertować filmu: {e}")
        print(f"Pozostawiono: {raw_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = common.load_classes()

    for model_name, imgsz, conf in CONFIGURATIONS:
        model = common.load_model(model_name)
        cmap = common.class_map(model, classes)

        render_video(
            model,
            model_name,
            imgsz,
            conf,
            classes,
            cmap
        )

    print("\nWszystkie konfiguracje zostały przetworzone!")


if __name__ == "__main__":
    main()
