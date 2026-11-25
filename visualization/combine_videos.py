
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe
import config

VIDEOS = config.VIDEOS

# Kolejność: lewy górny, prawy górny,
# lewy dolny, prawy dolny
INPUTS = [
    "yolo26s_960_conf035.mp4",
    "yolo26s_960_conf010.mp4",
    "yolo26s_640_conf030.mp4",
    "yolo26n_320_conf010.mp4",
]

OUTPUT = VIDEOS / "comparison_2x2.mp4"

# Każdy film zajmie 960x540 pikseli
# Wynik końcowy: 1920x1080
FILTER = (
    "[0:v]fps=30,setpts=PTS-STARTPTS,"
    "scale=1920:1080,setsar=1[v0];"
    "[1:v]fps=30,setpts=PTS-STARTPTS,"
    "scale=1920:1080,setsar=1[v1];"
    "[2:v]fps=30,setpts=PTS-STARTPTS,"
    "scale=1920:1080,setsar=1[v2];"
    "[3:v]fps=30,setpts=PTS-STARTPTS,"
    "scale=1920:1080,setsar=1[v3];"
    "[v0][v1][v2][v3]"
    "xstack=inputs=4:"
    "layout=0_0|w0_0|0_h0|w0_h0:"
    "fill=black[out]"
)


def main():
    for name in INPUTS:
        if not (VIDEOS / name).exists():
            raise FileNotFoundError(f"Brak filmu: {name}")

    command = [get_ffmpeg_exe(), "-y"]

    for name in INPUTS:
        command += ["-i", str(VIDEOS / name)]

    command += [
        "-filter_complex", FILTER,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
        "-shortest",
        "-movflags", "+faststart",
        str(OUTPUT),
    ]

    print("Łączenie czterech filmów...")
    subprocess.run(command, check=True)

    print(f"Gotowe! Zapisano: {OUTPUT}")


if __name__ == "__main__":
    main()
