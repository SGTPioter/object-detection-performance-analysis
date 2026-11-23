"""Krok 5: analiza fałszywych wykryć (FP) i pominiętych obiektów (FN).

Używa zapisanych predykcji z evaluate.py (nie uruchamia modelu ponownie).
Rysuje: zielone = trafienia (TP), czerwone = fałszywe wykrycia (FP, z typem błędu),
niebieskie = pominięte obiekty wzorcowe (FN). Zapisuje obrazy tylko z błędami
i wypisuje podsumowanie, które klasy najczęściej są gubione lub mylone.

Przykład:
    python visualize_errors.py --model yolo11n.pt --imgsz 640 --conf 0.5
"""
import argparse
import json
from collections import Counter

import cv2
import numpy as np

import common
import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=config.MODELS[0])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=config.REPORT_CONF)
    ap.add_argument("--max-images", type=int, default=60, help="ile obrazów z błędami zapisać")
    args = ap.parse_args()

    classes = common.load_classes()
    tag = f"{args.model.replace('.pt', '')}_{args.imgsz}"
    pfile = config.RESULTS / "predictions" / f"{tag}.json"
    if not pfile.exists():
        raise SystemExit(f"Brak {pfile}. Najpierw uruchom evaluate.py.")
    preds = json.loads(pfile.read_text())

    out = config.RESULTS / "errors" / f"{tag}_conf{args.conf:.2f}"
    out.mkdir(parents=True, exist_ok=True)
    fp_kinds, fn_cls, fp_cls = Counter(), Counter(), Counter()
    confusions = Counter()  # (klasa wzorca, klasa wykrycia) dla 'zła klasa'
    saved = 0

    for p in preds:
        img = cv2.imread(str(config.IMAGES / p["path"]))
        h, w = img.shape[:2]
        lab = config.LABELS / (config.Path(p["path"]).stem + ".txt")
        if not lab.exists():
            continue
        gb, gc = common.read_yolo_labels(lab, w, h)
        boxes = np.array(p["boxes"]).reshape(-1, 4)
        scores, cls = np.array(p["scores"]), np.array(p["cls"], int)
        keep = scores >= args.conf
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        m = common.match_image(gb, gc, boxes, scores, cls)

        for i, kind in m["fp"]:
            fp_kinds[kind] += 1
            fp_cls[classes[cls[i]]] += 1
            if kind == "zła klasa":
                ious = common.iou_matrix(boxes[i:i + 1], gb)[0]
                j = int(np.argmax(np.where(gc != cls[i], ious, -1)))
                confusions[(classes[gc[j]], classes[cls[i]])] += 1
        for j in m["fn"]:
            fn_cls[classes[gc[j]]] += 1

        if (m["fp"] or m["fn"]) and saved < args.max_images:
            canvas = img.copy()
            common.draw_boxes(canvas, [boxes[i] for i, _, _ in m["tp"]],
                              [f"{classes[cls[i]]} {scores[i]:.2f}" for i, _, _ in m["tp"]], common.GREEN)
            common.draw_boxes(canvas, [boxes[i] for i, _ in m["fp"]],
                              [f"FP {classes[cls[i]]} {scores[i]:.2f} ({k})" for i, k in m["fp"]], common.RED)
            common.draw_boxes(canvas, [gb[j] for j in m["fn"]],
                              [f"FN {classes[gc[j]]}" for j in m["fn"]], common.BLUE, 3)
            cv2.imwrite(str(out / p["path"]), canvas)
            saved += 1

    print(f"Konfiguracja: {args.model}, imgsz={args.imgsz}, conf={args.conf}")
    print(f"Zapisano {saved} obrazów z błędami w {out}\n")
    print("Typy fałszywych wykryć (FP):", dict(fp_kinds) or "brak")
    print("FP według klasy wykrycia   :", dict(fp_cls.most_common()) or "brak")
    print("FN według klasy wzorca     :", dict(fn_cls.most_common()) or "brak")
    if confusions:
        print("Pomyłki klas (wzorzec -> model):", {f"{a} -> {b}": n for (a, b), n in confusions.most_common()})
    print("\nNiebieskie ramki = pominięte obiekty. Sprawdź, czy to małe, zasłonięte lub rozmyte "
          "przedmioty,\na także czy czerwone ramki nie są w rzeczywistości brakami w Twoich etykietach.")


if __name__ == "__main__":
    main()
