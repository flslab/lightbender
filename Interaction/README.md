# Authors: 
Shuqin Zhu, Shahram Ghandeharizadeh

Description: 
* This directory contains the cage design for LightBenders and Crazyflies(see [Cage Design](Cage_Design/) directory.).
* Standalone uilities to process logs from Raspbarry Pi carried by LightBenders.
* Tools to generate line charts.
* Tools to identify a LightBender in a video recording, and overlay the speed dashboard next to it.

# Run:
Prior to run Interaction, generate the swarm manifest file ```swarm_manifest.yaml``` under orchestrator directory (see [example](../orchestrator/swarm_manifest_sample.yaml).
Once the file was generated and configured, run:
```bash
python3 "$(git rev-parse --show-toplevel)/orchestrator/orchestrator.py" --interaction --intractable-illumination
```
The following videos show:

- [Shahram and Shuqin interacting with a happy emoji.](https://youtu.be/_huysBvEuR4)
- [Vertical interaction with a LightBender.](https://youtu.be/ilh584zCPyQ)

Bare-hand interactions have diverse applications ranging from health-care to education and entertainment.  Our Blender add-on enables one or more users to adjust the position of LightBenders in an illumination using their bare-hands.  For a video demonstration, see [authoring](https://youtu.be/_I6qcD0NoYM).

# Implementation
Details of the authoring tool are available in [Authoring Tool](https://github.com/flslab/lightbender/tree/master/authoring).

The source code for interaction is available in [Inreaction Controller](https://github.com/flslab/fls-cf-offboard-controller/tree/master/Interaction) directory.
Further instructions to run this feature of LightBender, see [Controller README](https://github.com/flslab/fls-cf-offboard-controller/README.md).