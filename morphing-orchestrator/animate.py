import matplotlib
matplotlib.use("Agg")

import os
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

image_dir = "mission_visualizations"

# Determine frame number based on the following format: mission_frame_i.png, where i = frame #
def get_frame_number(filename):
    match = re.search(r"mission_frame_(\d+)\.png", filename)
    return int(match.group(1)) if match else -1

# Get files and sort numerically
files = sorted(
    [
        f for f in os.listdir(image_dir)
        if f.startswith("mission_frame_") and f.endswith(".png")
    ],
    key=get_frame_number
)


files = [os.path.join(image_dir, f) for f in files]

if not files:
    raise ValueError("No mission_frame_*.png images found.")

fig, ax = plt.subplots()
ax.axis("off")

img = mpimg.imread(files[0])
im = ax.imshow(img)

def update(frame):
    img = mpimg.imread(files[frame])
    im.set_array(img)
    return [im]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(files),
    interval=1000  # 1 second per frame
)

ani.save("mission_animation.mp4", writer="ffmpeg", fps=1)