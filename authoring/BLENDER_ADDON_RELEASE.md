# Blender add-on compatibility and release design

The add-on targets Blender 3.0 and later. Version 1.24.3 replaces the repository-
dependent single-file installation with a standalone legacy add-on ZIP. The
legacy format deliberately covers Blender 3.x/4.0–4.1 as well as newer releases;
a Blender 4.2-only extension manifest would exclude the older supported versions.

## Findings and fixes

| Area | Previous behavior | Release behavior |
| --- | --- | --- |
| Preferences identity | `__name__` was used for both preferences registration and lookup. This works for a properly installed single-file add-on, but the identity changes when packaged or imported as a submodule. Text Editor execution does not create an installed add-on entry. | Both use `__package__ or __name__`. Installation is a named package. Authoring no longer needs a repository setting; hardware settings remain visible in Preferences. |
| Animation | Many operators accessed `action.fcurves` directly. Blender 4.4's compatibility accessor addresses only the first slot; Blender 5 removes it. | All curve lookup, editing, and removal uses the object's assigned action slot/channelbag on 4.4+, with the original API on 3.0–4.3. Tests cover two objects sharing an Action in different slots. |
| Script discovery | Helpers were expected beside a copied `.py` file or in `<repository>/authoring`. | Every authoring helper is shipped in the ZIP and resolved relative to the installed package. |
| Python and platforms | Helpers used `python3`, a login `zsh`, shell quoting, or an unrelated repository venv. Hardware launch used AppleScript/Terminal. | Authoring uses the private venv interpreter and argument arrays. Hardware uses its explicitly configured environment and an interactive terminal adapter on each platform, retaining its camera/network confirmation prompts. |
| Dependency isolation | Availability depended on the user's shell and Python packages. | Background setup creates a private venv. Blender imports only standard-library runtime code. No changes to Blender's Python packages or `sys.path` are needed. |
| Plotting imports | `place.py` and `deconflict.py` selected macOS's plotting backend and imported visualization dependencies even with `--no_viz`. | Plotting imports are lazy. Default authoring needs only NumPy, SciPy, PyYAML, and svg.path. Gurobi is an optional setup profile. |
| Working files | Predictable global temp names could collide or reuse an old helper result. | Each operation gets a private temporary workspace; Stagger removes its previous frame's output before a new solve. |
| Rendering | MP4 export assumed every Blender build included FFmpeg and selected video using the legacy `file_format = "FFMPEG"` enum. | The operator checks FFmpeg support and uses `media_type = "VIDEO"` when available, retaining the legacy setting on older versions. Both foreground and generated background scripts produce H.264 MP4s. |
| Disable/re-enable | Monitor sockets and timers could outlive the registration lifecycle. | Unregistration closes both session types and removes setup/session callbacks before unregistering classes. Setup may finish as an independent child process after disabling. |

The existing preferences class was already in the registration tuple, and its
annotated `StringProperty(subtype='DIR_PATH')` is supported by Blender 3.0.
The reported missing repository field was not evidence that directory fields
are unsupported. The package fixes the identity ambiguity and removes that
setting from the required installation workflow. The legacy single-file add-on
must be disabled before enabling the package, since both register the same
operators and scene properties.

No personal absolute paths are embedded in the released code. Python and data
paths come from the installed package, Blender's per-user configuration path,
explicit user settings, or private temporary directories. Localhost endpoints
used for Blender/orchestrator communication remain intentional configuration
defaults.

## What the ZIP contains

```text
lightbender_swarm/
  __init__.py                  # generated from blender_addon.py
  addon_runtime.py
  bootstrap_runtime.py
  requirements-addon.txt
  requirements-addon-gurobi.txt
  README.md
  BLENDER_ADDON_RELEASE.md
  release_manifest.json
  authoring/
    transform.py
    place.py
    set_cover_gurobi_f.py
    deconflict.py              # Stagger
    morph_match.py
    camera_facing.py
```

The builder uses an explicit file allowlist, syntax checks, deterministic ZIP
metadata/order, and SHA-256 checksums. It does not traverse the checkout to
collect user data. Two identical source trees produce identical archives.

The orchestrator is a separate hardware deployment: it imports its own camera,
SSH, networking, calibration, and uploader components and uses a local hardware
manifest plus code deployed on drones. It is deliberately not a hidden
installation dependency of a graphics authoring add-on. All authoring and file
export operations work without it. Live-control buttons validate the hardware
repository/interpreter configuration before opening sessions or launching it.

## Environment installation and updates

1. Enabling schedules setup after registration, so installation does not hold
   the Preferences UI while pip downloads packages.
2. Setup prefers Blender's actual bundled Python executable, never the Blender
   application executable. An explicit Python 3.9+ override is available.
3. The environment directory is fingerprinted by interpreter identity/file
   metadata, operating system/architecture, dependency files, and optional solver
   selection. Changing these selects a fresh environment, leaving active ones
   usable by another Blender session.
4. A separate Python process uses `venv` and isolated pip with binary wheels.
   Python's inherited environment and user site are excluded; pip does not use
   user configuration or `PIP_TARGET`/`PIP_PREFIX` destination overrides.
5. A per-environment lock serializes concurrent setup. A readiness marker is
   written atomically only after dependency imports pass. Interrupted or corrupt
   markers and missing interpreters are repaired on retry. If a forcibly killed
   installer leaves a lock, the setup log identifies its path; after verifying no
   installer is running, remove that lock and retry.
6. Blender polls progress without importing scientific packages. Tools report
   setup status until ready. Newer Blender's online-access preference is
   respected. A ready environment works offline.

First setup requires internet access to obtain the dependency wheels. This is a
standalone code distribution, not an offline, multi-platform wheel bundle.
Requirements use bounded compatible ranges and pip selects versions with the
appropriate Python/OS wheels. The ZIP itself is reproducible; package versions
selected from those ranges may change between fresh installations. For fully
reproducible offline deployments, produce per-Python/per-platform wheel sets
and pinned constraints as a separate distribution profile.

After uninstalling, the private environment/cache remains in the per-user
configuration directory. It can be removed manually after closing sessions that
use it. Removing an add-on never deletes a shared data directory automatically.

## Validation and remaining release checks

Local validation on macOS now includes Blender 3.0.1 (bundled Python 3.9),
4.5.3 LTS (Python 3.11), and 5.2.3 LTS (Python 3.13). The
compatibility smoke covers real RNA registration/preferences, material creation,
Action slots, keyframe edits, and Flight Attitude. Existing marker-grid,
trajectory, and foreground/background video tests also run against the real Blender API.
The extracted ZIP also passed a complete authoring test with no repository
configured: real venv setup, package preference lookup/drawing, VFG and SC/bnb
placement, Morph, YAML import, Camera Facing, and Stagger/reset. The package was
installed in a path containing spaces/apostrophes. SC placement/Morph passed
with the package and process working directory made read-only; no files were
written there. Disable/re-enable removed and restored timer registration
correctly. All 27 runtime/ZIP unit tests and six camera-facing unit tests pass.
Runtime and ZIP tests cover isolation, offline reuse, failure/repair behavior,
package completeness, hashes, and reproducibility.

Verification on 2026-10-22:

| Headless check | Blender 3.0.1 | Blender 5.2.3 LTS |
| --- | --- | --- |
| Preferences/RNA, materials, Action editing | Pass | Pass |
| Defaults, module geometry, sequential drone IDs | Pass | Pass |
| Legacy marker-grid coordinates/timing | Pass | Pass |
| Web marker-grid coordinates/timing | Pass | Pass |
| Trajectory export | Pass | Pass |
| Actual foreground and background H.264 MP4 renders | Pass | Pass |
| Extracted standalone ZIP and fresh venv setup | Pass (Python 3.9) | Pass (Python 3.13) |
| VFG/SC placement, Morph, YAML import, Camera Facing, Stagger/reset | Pass | Pass |
| Read-only working directory, timer disable/re-enable cleanup | Pass | Pass |

The Blender 5.2.3 video test initially exposed removal of the `FFMPEG` image
format enum. Version 1.24.3 fixes both rendering paths by checking for the newer
`media_type` property. Foreground/background video regression checks also pass
on Blender 4.5.3 after the fix. Tests used disposable Blender processes without
saving user preferences or launching a physical swarm. Preferences fields were
validated through RNA and a simulated draw call; this was not a manual UI test.

Blender 3.6 and native Windows/Linux were not executed locally. The CI matrix
provides additional checks before release; its remote jobs have not yet run.
Hardware launch was not exercised against a real swarm.

The current repository has no applicable add-on/root license. Choose a license
before a public release; no license or extension-platform eligibility is assumed.
A future Blender 4.2+ extension can be generated as a second artifact, with a
manifest and platform-specific wheels, while retaining this legacy ZIP for
Blender 3.x.

## Primary API references

- [Blender AddonPreferences and package identity](https://docs.blender.org/api/4.4/bpy.types.AddonPreferences.html)
- [Blender 3.0 preferences and installed add-on collection](https://docs.blender.org/api/3.0/bpy.types.Preferences.html)
- [Blender 4.4 Action API changes and removal schedule](https://developer.blender.org/docs/release_notes/4.4/python_api/)
- [Blender 4.2 legacy and extension packaging](https://developer.blender.org/docs/release_notes/4.2/python_api/)
- [Installing legacy ZIP add-ons in newer Blender](https://docs.blender.org/manual/en/5.1/editors/preferences/addons.html)
- [pip isolation options](https://pip.pypa.io/en/stable/cli/pip/)

- [Blender 5 video media type](https://docs.blender.org/api/5.0/bpy.types.ImageFormatSettings.html)
