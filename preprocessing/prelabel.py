"""Krok 2 (opcjonalny): wstępne etykiety do RĘCZNEJ poprawy.

Duży model zaznacza obiekty, a Ty poprawiasz je w narzędziu do etykietowania
(CVAT, Label Studio, LabelImg). Poprawianie jest dużo szybsze niż rysowanie od zera.

UWAGA METODOLOGICZNA: jeśli zostawisz ramki bez sprawdzenia, ocenisz modele względem innego
modelu, a nie względem prawdy. Przejrzyj KAŻDĄ klatkę: usuń błędne ramki, dodaj pominięte,
popraw krzywe. Domyślnie używamy RT-DETR (inna architektura niż badane YOLO), aby nie
faworyzować badanej rodziny modeli. Opisz to w raporcie.

Przykład:
    python prelabel.py --model rtdetr-l.pt --conf 0.35
"""
import argparse

import cv2

import common
import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--overwrite", action="store_true", help="nadpisz istniejące etykiety")
    args = ap.parse_args()

    classes = common.load_classes()
    model = common.load_model(args.model)
    cmap = common.class_map(model, classes)
    config.LABELS.mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    for path in common.list_images():
        out = config.LABELS / f"{path.stem}.txt"
        if out.exists() and not args.overwrite:
            continue
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        boxes, _, cls, _ = common.predict(model, img, args.imgsz, args.conf, cmap)
        common.write_yolo_labels(out, boxes, cls, w, h)
        n_img += 1
        n_box += len(boxes)
    print(f"Zapisano etykiety dla {n_img} obrazów ({n_box} ramek) w {config.LABELS}.")
    print("TERAZ POPRAW JE RĘCZNIE (format YOLO, kolejność klas jak w classes.txt).")


if __name__ == "__main__":
    main()
