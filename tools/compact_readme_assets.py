"""Create whitespace-cropped GIF variants for the root README."""

from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "01_RRT_Star_Karaman/results/rrt2500.gif",
    "01_RRT_Star_Karaman/results/rrta2500.gif",
    "01_RRT_Star_Karaman/results/rrta10000.gif",
    "02_PID_Kanayama/results/kanayama_demo.gif",
    "03_MPC_Kong/results/success.gif",
    "03_MPC_Kong/results/drift.gif",
    "04_DQN_Minh/results/progress_50M.gif",
    "06_PPO_Schulman/results/breakout_gameplay.gif",
]


def content_box(image: Image.Image, threshold: int = 8, padding: int = 10) -> tuple[int, int, int, int]:
    """Return one padded content box that is valid for every animation frame."""
    union = None
    for frame_index in range(image.n_frames):
        image.seek(frame_index)
        frame = image.convert("RGB")
        background = Image.new("RGB", frame.size, frame.getpixel((0, 0)))
        mask = ImageChops.difference(frame, background).convert("L")
        mask = mask.point(lambda value: 255 if value > threshold else 0)
        box = mask.getbbox()
        if box is None:
            continue
        if union is None:
            union = box
        else:
            union = (
                min(union[0], box[0]),
                min(union[1], box[1]),
                max(union[2], box[2]),
                max(union[3], box[3]),
            )

    if union is None:
        return (0, 0, image.width, image.height)

    return (
        max(0, union[0] - padding),
        max(0, union[1] - padding),
        min(image.width, union[2] + padding),
        min(image.height, union[3] + padding),
    )


def compact(source: Path) -> Path:
    image = Image.open(source)
    box = content_box(image)
    source_size = image.size
    image.close()

    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    destination = source.with_name(f"{source.stem}_compact.gif")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to preserve GIF delta-frame compression")

    filter_graph = (
        f"crop={width}:{height}:{left}:{top},split[c0][c1];"
        "[c0]palettegen=stats_mode=diff[p];"
        "[c1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
    )
    print(f"{source.relative_to(ROOT)}: {source_size} -> {(width, height)}")
    return destination


if __name__ == "__main__":
    for relative_source in SOURCES:
        compact(ROOT / relative_source)
