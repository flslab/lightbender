
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

# Path to your image directory
image_dir = "mission_visualizations"

# Get sorted list of image files
files = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith((".png", ".jpg", ".jpeg"))
])


fig, ax = plt.subplots()
ax.axis("off")  # hide axes

# Load first image
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
    interval=1000  # milliseconds between frames
)

plt.show()
