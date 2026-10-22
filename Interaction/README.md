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

For momentum-model identification, run the separate contact-free flow:

```bash
python3 "$(git rev-parse --show-toplevel)/orchestrator/orchestrator.py" --calibrate
```

This stores per-axis delay, first-order time constant, and acceleration scale
on each drone in `Interaction/wrench_calibration.json`. Normal `--interaction`
runs load that file automatically and never issue the calibration excitation.
Plain calibration runs only the XYZ wrench-model stage. Add
`--adaptive-braking-calibration` to explicitly run the model-guided planar
attitude/braking stage as well.
Prediction-model validation and legacy planar-fit validation are independent.
An adaptive run may retain an existing valid planar fit while saving a newly
validated prediction model; without an existing valid planar fit, first run the
fixed baseline opt-out above.
Adaptive candidates are eligible only when the predicted horizon endpoint is
within 0.05 m/s of zero projected velocity and 3 degrees of level attitude.
The `translation_inertia` mission currently uses fourteen paired-direction
attitude trials. Its 0.16/0.24/0.32-second tiers are retained and the new
0.45-second tier is repeated three times so measured speeds near 1.5 m/s can
enter training, receive an independent held-out validation, and only then be
used by an adaptive trial. Because this changes the exact calibration protocol,
collect a new fixed baseline first:

```bash
python3 "$(git rev-parse --show-toplevel)/orchestrator/orchestrator.py" \
  --calibrate --no-adaptive-braking-calibration
```

Do not reuse the earlier 0.32-second planar fit as evidence that the new speed
tier is covered; confirm the measured brake-entry range in the resulting log.
The following videos show:

- [Shahram and Shuqin interacting with a happy emoji.](https://youtu.be/_huysBvEuR4)
- [Vertical interaction with a LightBender.](https://youtu.be/ilh584zCPyQ)

Bare-hand interactions have diverse applications ranging from health-care to education and entertainment.  Our Blender add-on enables one or more users to adjust the position of LightBenders in an illumination using their bare-hands.  For a video demonstration, see [authoring](https://youtu.be/_I6qcD0NoYM).

# Implementation
Details of the authoring tool are available in [Authoring Tool](https://github.com/flslab/lightbender/tree/master/authoring).

The source code for interaction is available in [Inreaction Controller](https://github.com/flslab/fls-cf-offboard-controller/tree/master/Interaction) directory.
Further instructions to run this feature of LightBender, see [Controller README](https://github.com/flslab/fls-cf-offboard-controller/README.md).
