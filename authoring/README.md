# LightBender Authoring

This directory contains the tools for designing, animating, and visualizing LightBender drone swarms. The primary authoring interface is a Blender add-on that enables an artist to place LightBenders in 3D space, animate LEDs and servos, and export an SFL (mission YAML) file for physical illumination using LightBenders and the orchestrator.  The add-on supports a novel edit mode that enables the artist to interact with the LightBenders and adjust their location using their bare hand. 

Authors: Hamed Alimohammadzadeh(halimoha@usc.edu), Shuqin Zhu (shuqinzh@usc.edu), and Shahram Ghandeharizadeh (shahram@usc.edu)

---

## Edit Mode Using Bare-Hand Interactions

We support bare-hand interactions with LightBenders to edit graphics.  Multiple users may adjust the position of LightBenders using their bare hand simultaneously.  Our implementation of the Blender add-on updates the data in real-time.  The following video shows the workflow of a bare-hand interaction in support of edit mode. 

- [Blender add-on in edit mode with bare-hand interactions.](https://youtu.be/_I6qcD0NoYM)

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
- **Generative layouts** — automatically place LightBenders along SVG strokes using Vertex First Greedy (VFG) or Set Cover (SC) placement algorithms
- **LED animation** — define LED pointers (split points + color expressions) and apply global color expressions
- **Global LED effects** — apply procedural animation presets (Static Color, Blinking, Sparkle, Rainbow Wheel, Color Chase, Pulse, Color Cycle) to all or selected LightBenders
- **Draw/Erase effect** — sequence or simultaneously animate LightBenders writing strokes, with per-drone direction control and keyframe reversal
- **Fold effect** — keyframe LightBenders unfolding from a straight lit state into their target pose
- **Fly-In/Fly-Out effect** — generate exploded-assembly fly-in sequences
- **Morph** — transition the swarm from the current SVG layout to a new SVG shape
- **Position errors & drift** — inject random positional noise or continuous drift for simulation realism
- **Stagger** — stagger LightBender positions based on camera perspective to avoid downwash
- **Live swarm monitoring** — real-time per-drone status and battery voltage in list or grid view with filtering
- **Export & launch** — write the full animation to YAML and optionally launch the orchestrator or an interactive bare-hand editing session directly

### Setting `REPO_DIR` and Installing the Add-On

1. Open `blender_addon.py` and set `REPO_DIR` at the top of the file to the absolute path of your local repository clone:

   ```python
   REPO_DIR = "/path/to/lightbender"
   ```

2. In Blender, go to **Edit > Preferences > Add-ons > Install…**, select `blender_addon.py`, and enable the **LightBender Swarm Animator** add-on.

3. The panel will appear in the **3D Viewport sidebar** under the **LightBender** tab (`N` key to open).

---

## Panel Reference

The sidebar is organized into one root panel and five collapsible sub-panels.

---

### LightBender Controls

The root panel. Always visible.

#### Add LightBender

Select a **Light Element Type** from the drop-down, then click **Create**:

| Type | Description |
|---|---|
| **Type H (Rod)** | Two-segment rod. Segment 1 spans 0–180°, Segment 2 spans 180–360°. |
| **Type V (Rod)** | Two-segment rod rotated 90°. Segment 1 spans 90–270°, Segment 2 spans 270–450°. |
| **Semicircle H** | Single-segment arc, 0–180°. One servo. |
| **Semicircle V** | Single-segment arc, 90–270°. One servo. |
| **Ring** | Full circle, no servo actuation. |

When **Ring** is selected, a **Ring Radius** (m) field appears before the Create button.

#### Selected LightBender Controls

These controls appear when a LightBender object (`lb*`) is the active selection.

**Servo Angles**

- **Angle 1** (or **Arc Angle** for Semicircle types) — servo 1 angle in degrees.
- **Angle 2** — servo 2 angle in degrees (hidden for Semicircle types).
- Ring types show a "No Actuation" label; servo fields are suppressed.

**LED Configuration**

- **Mode** — choose between two per-drone LED modes:
  - **Single Expression** — one Python formula evaluated per LED. Available variables: `i` (LED index), `t` (time in seconds), `N` (total number of LEDs), `math`, and `random`. Example: `[255, 255, 255] if i < t*10 else [0, 0, 0]`.
  - **Pointers** — define colored segments using a base color and a list of split-point pointers. Each pointer has a position value and a color expression; LEDs before a pointer inherit the previous segment's color.
    - **Base Color (Start)** — color expression applied before the first pointer.
    - **Pointers list** — add/remove pointers with `+`/`−`. Each row shows the split position and the color to the right of that split.

- **Test/Update LEDs** — re-evaluates all LED formulas and refreshes viewport colors immediately (useful after editing formulas without scrubbing the timeline).

---

### Generative Layouts

Automates swarm placement from an SVG file.

#### SVG Transform and Place

Reads an SVG shape and a swarm manifest and places LightBenders along the shape's strokes.

| Field | Description |
|---|---|
| **SVG File** | Path to the source SVG file. |
| **Swarm Manifest** | Path to the YAML manifest that lists available drones and their types. |
| **Max Width / Max Height** | Bounding box (m) to scale the SVG layout into. |
| **Center X / Center Z** | World-space coordinates for the layout center. |
| **Placement Policy** | **VFG** or **Set Cover** — algorithm used to place drones along the shape's strokes. |
| **LED Color** | Python RGB list (e.g. `[255, 0, 0]`) applied to all placed drones. |

Click **Transform and Place** to run the placement.

---

### Automated Animations

Contains four keyframe-generation tools.

#### Morph

Transitions the swarm from its current SVG-based layout to a second shape.

| Field | Description |
|---|---|
| **Shape 2 SVG** | Path to the target SVG file. |
| **Hold 1 (s)** | Time to hold the initial shape before moving. |
| **Morph (s)** | Duration of the transition between shapes. |
| **Hold 2 (s)** | Time to hold the final shape after arriving. |

Click **Generate Morph** to insert keyframes.

#### Fold

Animates all selected LightBenders unfolding from a straight, fully-lit state into their current posed angles.

| Field | Description |
|---|---|
| **Fold Angle (deg)** | Servo-1 angle for the straight state (Servo 2 = Fold Angle + 180°). TYPE_V offsets this by 90°. |
| **Hold (s)** | Duration of the straight state before folding. |
| **Transition (s)** | Duration of the fold-in motion. |
| **Align** | Snap all LBs to the swarm centroid along the selected axis (X/Y/Z) during the straight state. |
| **Distribute** | Evenly space all LBs along the selected axis (X/Y/Z) during the straight state. |
| **Equalize Size** | Scale each LB's LED pointer range by its camera distance so all rods appear the same on-screen length. |
| **Resolve Overlaps** | After positioning, stretch LB positions along the selected axis until the minimum inter-LB gap equals **Min Spacing**. |
| **Min Spacing (m)** | Minimum allowed distance between any two LBs when overlap resolution is active. |

Click **Generate Fold** to write keyframes; **Trash** icon clears them.

#### Draw/Erase

Sequences LightBenders to illuminate progressively, simulating a stroke being drawn and then erased.

**Groups list** — each group represents one stroke or letter. Groups can be Sequential (drawn one after the other) or Simultaneous (drawn in parallel). Use `+`/`−` to add/remove groups; click a group to manage its members.

**LightBenders in \<group\>** — ordered list of drones assigned to the active group. Use `+`/`−` to add/remove drones (adds the current viewport selection); arrow buttons reorder them.

**Settings:**

| Field | Description |
|---|---|
| **Draw Speed (LEDs/s)** | Rate at which LEDs illuminate during the draw pass. |
| **Erase Speed (LEDs/s)** | Rate at which LEDs turn off during the erase pass. |
| **Hold Time (s)** | Pause between the end of draw and the start of erase. |
| **Overlap Next (%)** | How much the next group's draw begins before the current group finishes (0 = fully sequential, 100 = fully overlapping). |
| **Draw RGB** | Color used during the draw pass (Python RGB list). |
| **Erase RGB** | Color used during the erase pass / background color (Python RGB list). |

Click **Generate Draw/Erase** to write keyframes; **Trash** icon clears them.

#### Fly-In / Fly-Out

Generates an exploded-assembly animation: drones fly in from off-screen positions to their targets, hold, then fly back out.

| Field | Description |
|---|---|
| **Ref Distance (m)** | Distance of the explosion reference point behind the centroid. |
| **Min/Max X, Y, Z (m)** | World-space bounding box from which start positions are sampled. |
| **Outward Duration (s)** | Time for drones to travel from their targets to the exploded positions. |
| **Hold Duration (s)** | Time drones pause at the exploded positions. |
| **Inward Duration (s)** | Time for drones to fly from the exploded positions back to their targets. |

Click **Generate Fly-In/Fly-Out** to write keyframes; **Trash** icon clears them.

---

### Global LED Effects

Applies a procedural LED animation preset to all selected (or all) LightBenders in the scene.

**Preset** — choose from seven animation presets:

| Preset | Description |
|---|---|
| **Static Color** | Single, unchanging color. |
| **Global Blinking** | All LEDs cycle through the color list simultaneously. |
| **Random Sparkle** | Each LED randomly flickers independently. Controlled by **Sparkle Threshold**. |
| **Rainbow Wheel** | A moving rainbow gradient sweeps across LEDs over time. |
| **Color Chase** | A moving block of color travels along the LED strip. |
| **Pulse** | Brightness smoothly pulses in and out. |
| **Color Cycle** | Smoothly blends between two colors using a sin/cos oscillation. |

**Color list** — add/remove colors with `+`/`−`. Click **Load Preset Colors** to populate the list with sensible defaults for the selected preset.

| Field | Description |
|---|---|
| **Speed Multiplier** | Scales the animation speed for all presets. |
| **Sparkle Threshold** | (Sparkle only) Probability above which a LED shows the sparkle color. 0 = always on, 1 = never. |

Click **Apply to Selected** to write the expression only to the selected LightBenders, or **Apply to All** to target every LightBender in the scene.

---

### Physics & Simulation

Injects positional noise and staggers LightBender positions based on camera perspective to avoid downwash and overlaps.

#### Error & Drift Simulators

**Static Error (YZ Plane)**

Randomly offsets each LightBender to simulate real-world positioning inaccuracy.

| Field | Description |
|---|---|
| **Error Distance (m)** | Maximum random offset applied to each drone. |

Click **Apply Static Error** to offset positions; **Reset Static** to undo.

**Dynamic Drift (XYZ)**

Applies a smooth, sinusoidal drift to each LightBender's position over time, simulating hovering instability.

| Field | Description |
|---|---|
| **Max Drift XY (m)** | Maximum drift amplitude in the X and Y axes. |
| **Max Drift Z (m)** | Maximum drift amplitude in the Z axis. |
| **Wave Scale (Slower=Higher)** | Noise frequency scale; higher values produce slower, wider drift. |

Click **Apply Drift** to generate drift keyframes; **Reset Drift** to remove them.

#### Stagger

Resolves downwash and overlap conflicts by shifting LightBenders toward or away from the camera.

| Field | Description |
|---|---|
| **Overlap Threshold (m)** | Distance below which two drones are considered overlapping. |
| **Downwash Threshold (m)** | Horizontal distance below which a drone is in the downwash of an upper drone. |
| **Selection Method** | Algorithm for choosing which Lightbenders to stagger: **Greedy Max Degree**, **Greedy Top Z**, **Greedy Bottom Z**, **Brute Force**, or **Random**. |
| **Resolution Order** | Priority for resolving conflicts: **Max Degree**, **Top Z**, **Bottom Z**, or **Random**. |
| **Trajectory Type** | **Line of Sight** — move along the camera ray; **Global** — move along the global centroid. |
| **Move Direction** | **Away** from camera, **Towards** camera, or **Hybrid** (choose the direction that minimizes distance). |

Click **Stagger** to apply; **Reset Stagger** to restore original positions.

---

### Mission Export

Writes the animation to an SFL (YAML) file and optionally launches the orchestrator.

#### Export Config

| Field | Description |
|---|---|
| **Mission Name** | Base name used for the output SFL file (defaults to the Blender scene name). |
| **Dark Room** | Enable dark-room mode in the generated mission. |

#### Mode

Toggle between two export/illuminate modes:

**Illumination** 

| Field | Description |
|---|---|
| **Export Keyframes Only** | When enabled, only keyframe times are exported (variable `dt` per waypoint). When disabled, waypoints are sampled at a fixed rate. |
| **Export Rate (Hz)** | Sampling frequency when not in keyframe-only mode. |

Buttons:
- **Export YAML** — write the mission file to `orchestrator/SFL/<mission_name>.yaml`.
- **Illuminate** — export and immediately launch the orchestrator in a new terminal tab.
- **Confirm Launch** — enabled once all drones report `ready`; arms the swarm and begins the mission.
- **Stop** — commands all drones to land and waits for confirmation.

**Interaction**

| Field | Description |
|---|---|
| **S_D (mm/s)** | Speed threshold exported as `delta_v` (m/s) in the interaction YAML. |
| **Grace Time (s)** | Delay before the interaction controller begins accepting commands. |
| **Interaction Duration (s)** | Maximum duration of the interactive editing session. |

Buttons mirror Illumination mode, plus:
- **Edit** — opens a bare-hand editing session.
- **Finish Edit** — closes the editing session and returns drones to hover.

#### LightBender Status

Live swarm monitor, nested inside Mission Export. Updated in real time while the orchestrator is running.

**View Mode** — switch between **List** and **Grid** displays:

- **List** — shows all drones in a scrollable list with ID, status, and battery voltage. Filter controls:
  - **Status filter** — show All, Idle, Booting, Ready, or Landed drones.
  - **Voltage filter** — show All, Charged (≥ threshold), or Discharged (< threshold) drones.
  - **Battery Threshold (V)** — voltage cutoff for the charged/discharged filter (default 3.9 V).
  - Clicking a list entry selects the corresponding LightBender object in the 3D viewport.

- **Grid** — compact tile view with one button per drone labeled with its numeric ID. Tile color indicates status: gray = idle, yellow = booting, green = ready, blue = landed, red = error. Clicking a tile selects the drone in the 3D viewport.
