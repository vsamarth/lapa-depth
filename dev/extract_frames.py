import argparse
import subprocess
from pathlib import Path


def convert_video_to_frames(
    input_dir: Path,
    output_dir: Path,
    fps: float = 5.0,
    quality: int = 5,
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    """
    Convert Something-Something V2 .webm videos into frame folders for LAPA.

    Output structure:
        output_dir/
            1/
                img0001.jpg
                img0002.jpg
            2/
                img0001.jpg
                img0002.jpg
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        input_dir.glob("*.webm"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    if not videos:
        raise FileNotFoundError(f"No .webm files found in: {input_dir}")

    if limit is not None:
        videos = videos[:limit]

    print(f"Found {len(videos)} videos")
    print(f"Extract FPS: {fps}")
    print(f"JPEG quality: {quality}")
    print()

    for idx, video_path in enumerate(videos, start=1):
        video_id = video_path.stem
        save_dir = output_dir / video_id
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = list(save_dir.glob("img*.jpg"))
        if existing and not overwrite:
            print(f"[{idx}/{len(videos)}] skip {video_id} (already extracted)")
            continue

        output_pattern = str(save_dir / "img%04d.jpg")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y" if overwrite else "-n",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-qscale:v", str(quality),
            output_pattern,
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"[{idx}/{len(videos)}] done {video_id}")
        except subprocess.CalledProcessError as e:
            print(f"[{idx}/{len(videos)}] fail {video_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Something-Something V2 .webm videos to frame folders for LAPA."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .webm videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frame extraction FPS. For LAPA, 5 is a good starting point."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=5,
        help="JPEG quality for ffmpeg. Lower is higher quality."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N videos for testing"
    )

    args = parser.parse_args()

    convert_video_to_frames(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        quality=args.quality,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
