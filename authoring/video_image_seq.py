import os
import cv2
import argparse
import subprocess
from glob import glob


def parse_resolution(res_str):
    try:
        w, h = map(int, res_str.lower().split('x'))
        return w, h
    except:
        raise argparse.ArgumentTypeError("Resolution must be WIDTHxHEIGHT (e.g., 1920x1080)")


def load_images(directory):
    return sorted(glob(os.path.join(directory, "*.png")))


def build_sequence(image_files, total_frames, pingpong):
    if len(image_files) == 1:
        return [image_files[0]] * (total_frames if total_frames is not None else 1)

    forward = image_files

    if pingpong:
        reverse = image_files[-2:0:-1]
        full = forward + reverse
    else:
        full = forward

    if total_frames is None:
        return full

    seq = []
    i = 0
    while len(seq) < total_frames:
        seq.append(full[i % len(full)])
        i += 1

    return seq


def main():
    parser = argparse.ArgumentParser(description="Create browser-compatible MP4 from PNG sequence")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--name", default="output")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--resolution", type=parse_resolution, default="1920x1080")
    parser.add_argument("--pingpong", action="store_true")

    args = parser.parse_args()

    width, height = args.resolution

    image_files = load_images(args.dir)
    if not image_files:
        raise ValueError("No PNG files found.")

    total_frames = args.fps * args.duration if args.duration is not None else None

    sequence = build_sequence(image_files, total_frames, args.pingpong)

    output_path = os.path.join(args.dir, f"{args.name}.mp4")

    # FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(args.fps),
        "-i", "-",  # stdin
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # important for web playback
        output_path
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for path in sequence:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read {path}")

        img = cv2.resize(img, (width, height))
        process.stdin.write(img.tobytes())

    process.stdin.close()
    process.wait()

    print(f"Browser-compatible video saved to: {output_path}")


if __name__ == "__main__":
    main()
