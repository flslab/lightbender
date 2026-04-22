# LightBender Authoring

This directory contains the tools for designing, animating, and visualizing LightBender drone swarms. The primary authoring interface is a Blender add-on that lets you place LightBenders in 3D space, animate LEDs and servos, and export directly to  SFL (mission YAML) for physical illumination using LightBenders and the orchestrator.

---

## LightBender Visual Effects

LightBender visual effects for different illuminations:

- [Visual effects with ACM, LA, and NSF.](https://youtu.be/jwrJeGCwS_M)  
- [ACM Illumination.](https://youtu.be/oepzLSos4Xo)  
- [LA Illumination.](https://youtu.be/p_fx22tXJf8)  
- [NSF Illumination.](https://youtu.be/o2t5xLuE0Hw)  

## Blender Add-On

![LightBender Swarm Animator](img/blender_add_on.png)

**LightBender Swarm Animator** (`blender_addon.py`) is a Blender 3.0+ add-on located in `View3D > Sidebar > LightBender`. It provides a full GUI workflow for:

- **LightBender placement** — add Type H/V rods, semicircle, or ring light elements with correct geometry and LED strips
- **LED animation** — define LED pointers (split points + color expressions) and apply global color expressions
- **Draw/Erase effect** — sequence or simultaneously animate LightBenders writing strokes, with per-drone direction control and keyframe reversal
- **Fold effect** — keyframe LightBenders unfolding from a straight lit state into their target pose
- **Fly-In/Fly-Out effect** — generate exploded-assembly fly-in sequences
<!-- - **Morph** — transition the swarm from the current layout to a new SVG shape -->
- **Position errors & drift** — inject random positional noise or continuous drift for simulation realism
- **Stagger** — stagger LightBender positions based on camera perspective to avoid downwash
- **Export** — write the full animation to YAML and optionally launch the orchestrator directly

### Setting `REPO_DIR` and Installing the Add-On

1. Open `blender_addon.py` and set `REPO_DIR` at the top of the file to the absolute path of your local repository clone:

   ```python
   REPO_DIR = "/path/to/lightbender"
   ```

2. In Blender, go to **Edit > Preferences > Add-ons > Install…**, select `blender_addon.py`, and enable the **LightBender Swarm Animator** add-on.

3. The panel will appear in the **3D Viewport sidebar** under the **LightBender** tab (`N` key to open).

---




