"""Dependency-free launch argument helpers for interaction missions."""


def validate_interaction_modes(args):
    """Reject ambiguous experiment ownership before opening any connections."""
    modes = ('interaction', 'calibrate', 'braking_test')
    if sum(bool(getattr(args, mode, False)) for mode in modes) > 1:
        raise ValueError(
            '--interaction, --calibrate and --braking-test are mutually exclusive'
        )
    if getattr(args, 'braking_test', False) and getattr(args, 'sense', False):
        raise ValueError('--braking-test does not use --sense; omit --sense')
    if getattr(args, 'braking_test', False):
        for mode in ('illumination', 'intractable_illumination', 'morphing', 'blender'):
            if getattr(args, mode, False):
                raise ValueError('--braking-test cannot be combined with --' + mode.replace('_', '-'))


def interaction_application_flag(args):
    """Select the single controller experiment mode (legacy default is interaction)."""
    validate_interaction_modes(args)
    if getattr(args, 'braking_test', False):
        return '--braking-test'
    if getattr(args, 'calibrate', False):
        return '--calibrate'
    return '--interaction'


def interaction_mocap_args(drone, wrench_mode=False):
    """Build interaction mocap arguments using the pre-wrench defaults.

    ``wrench_mode`` remains in the signature for caller compatibility. Onboard
    wrench detection does not require mocap orientation, so a drone with an
    initial marker position defaults to pointcloud localization as before.
    """
    position = drone.get('init_pos')
    object_name = drone.get('obj_name', drone.get('label'))
    localization_mode = drone.get(
        'interaction_localization',
        'pointcloud' if position else 'rigidbody',
    )
    if localization_mode == 'pointcloud':
        if not position:
            raise ValueError(f"Drone {drone['id']} requires init_pos for pointcloud localization")
        return (
            f"--init-pos {position[0]} {position[1]} {position[2]} "
            "--vicon-mode pointcloud "
        )
    if localization_mode == 'rigidbody':
        if not object_name:
            raise ValueError(
                f"Drone {drone['id']} requires obj_name or label for rigidbody localization"
            )
        return (
            f"--obj-name {object_name} --vicon-mode rigidbody --vicon-full-pose "
        )
    raise ValueError(
        f"Unsupported interaction_localization for {drone['id']}: {localization_mode}"
    )
