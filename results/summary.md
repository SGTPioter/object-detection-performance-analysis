# Podsumowanie eksperymentu

Dopasowanie: IoU >= 0.5, jedna adnotacja = najwyżej jedno TP. Dla każdej konfiguracji pokazano próg pewności o najwyższym F1.

| name          |   conf |   precision |   recall |   f1 |   mean_iou_tp |   fp |   fn |   mean_ms |   p95_ms |   fps_pipeline |
|:--------------|-------:|------------:|---------:|-----:|--------------:|-----:|-----:|----------:|---------:|---------------:|
| yolo11n · 320 |   0.1  |        65.8 |     59.2 | 62.4 |         0.939 |   55 |   73 |    16.177 |   17.737 |         25.972 |
| yolo11n · 640 |   0.1  |        75.7 |     76.5 | 76.1 |         0.969 |   44 |   42 |    33.749 |   37.12  |         19.053 |
| yolo11n · 960 |   0.1  |        78.8 |     83.2 | 81   |         0.97  |   40 |   30 |    59.09  |   63.13  |         13.1   |
| yolo11s · 320 |   0.35 |        94.3 |     73.7 | 82.8 |         0.954 |    8 |   47 |    26.466 |   28.564 |         19.737 |
| yolo11s · 640 |   0.35 |        97   |     73.2 | 83.4 |         0.974 |    4 |   48 |    63.956 |   68.324 |         12.312 |
| yolo11s · 960 |   0.3  |        94.2 |     81.6 | 87.4 |         0.975 |    9 |   33 |   123.186 |  131.623 |          6.943 |
| yolo26n · 320 |   0.1  |        73   |     62   | 67.1 |         0.938 |   41 |   68 |    17.028 |   18.955 |         27.949 |
| yolo26n · 640 |   0.2  |        84.1 |     70.9 | 77   |         0.96  |   24 |   52 |    34.382 |   42.75  |         19.138 |
| yolo26n · 960 |   0.1  |        77.3 |     83.8 | 80.4 |         0.955 |   44 |   29 |    55.571 |   59.209 |         13.249 |
| yolo26s · 320 |   0.2  |        87.8 |     80.4 | 84   |         0.952 |   20 |   35 |    27.417 |   29.482 |         20.952 |
| yolo26s · 640 |   0.3  |        92.1 |     84.9 | 88.4 |         0.976 |   13 |   27 |    62.918 |   67.381 |         12.05  |
| yolo26s · 960 |   0.35 |        93.2 |     84.4 | 88.6 |         0.975 |   11 |   28 |   137.145 |  145.988 |          6.874 |

**Najwyższa jakość:** yolo26s · 960, conf 0.35 (F1 88.6%, precyzja 93.2%, czułość 84.4%, 137.1 ms/obraz).
**Czas rzeczywisty (>= 25 FPS):** yolo26n · 320, conf 0.10 (F1 67.1%, 27.9 FPS całego potoku).

> Próg pewności wybrano na tych samych danych, na których liczono metryki, więc wynik jest nieco optymistyczny. Zaznacz to w raporcie.