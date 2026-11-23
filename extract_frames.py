"""Krok 1: wycinanie klatek z nagrania do oznaczenia.

Sąsiednie klatki są prawie identyczne, więc bierzemy je w odstępach czasu
(domyślnie co 0,5 s), aby zbiór oceny nie był sztucznie zawyżony.

Przykład:
    python extract_frames.py --video data/video/biurko.mp4 --every-sec 0.5
"""
import argparse

import cv2

import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", default=str(config.VIDEO))
    ap.add_argument("--out", default=str(config.IMAGES))
    ap.add_argument("--every-sec", type=float, default=0.5, help="odstęp między klatkami [s]")
    ap.add_argument("--max-frames", type=int, default=120, help="górny limit liczby klatek")
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="pomiń rozmazane klatki (wariancja Laplasjanu; np. 60). 0 = wyłączone")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Nie można otworzyć filmu: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps * args.every_sec))
    out = config.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    saved = skipped = idx = 0
    while saved < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp >= args.min_sharpness:
                cv2.imwrite(str(out / f"frame_{saved:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            else:
                skipped += 1
        idx += 1
    cap.release()
    print(f"Film: {fps:.1f} FPS, {idx} klatek. Zapisano {saved} klatek w {out} (pominięto rozmazanych: {skipped}).")
    print("Dalej: oznacz ramki (CVAT / Roboflow / LabelImg) albo uruchom prelabel.py.")


if __name__ == "__main__":
    main()
