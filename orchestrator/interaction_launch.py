"""Dependency-free launch argument helpers for interaction missions."""

import shlex
from copy import copy


def resolve_interaction_sensing(args, config):
    """Resolve one mission's sensor requirement without mutating CLI arguments.

    For level_coast, the detector or ``record_potentiometer`` requests acquisition;
    legacy ``sensing`` and ``--sense`` overrides are ignored. Separate hardware
    experiments still require their sensor. Standard interactions retain their
    existing optional ``sensing`` / CLI controls.
    Calibration/baseline/hover never inherit interaction sensor requirements.
    """
    resolved = copy(args)
    validate_interaction_modes(resolved, defer_sensing=True)
    other_mode = any(getattr(resolved, name, False) for name in (
        'calibrate', 'braking_test', 'mpc', 'baseline', 'hover',
        'active_septic_brake_calibration'))
    if not other_mode and (getattr(resolved, 'interaction', False)
                           or getattr(resolved, 'sense', False)):
        if not isinstance(config, dict):
            raise ValueError('Interaction.config must be a mapping')
        record_potentiometer = config.get('record_potentiometer', False)
        if type(record_potentiometer) is not bool:
            raise ValueError('Interaction.config.record_potentiometer must be boolean')
        if config.get('behavior') == 'level_coast':
            options = config.get('level_coast') or {}
            if not isinstance(options, dict):
                raise ValueError('Interaction.config.level_coast must be a mapping')
            if 'detector' in options:
                raise ValueError('level_coast.detector was removed; use config.detection_method')
            detector = config.get('detection_method', 'potentiometer')
            if detector not in ('potentiometer', 'model', 'vel'):
                raise ValueError('level_coast detection_method must be potentiometer, model or vel')
            required = detector == 'potentiometer'
        else:
            virtual = config.get('virtual_object') or {}
            contact = virtual.get('contact_detection') or {}
            release = virtual.get('release_behavior') or {}
            required = (contact.get('source') == 'potentiometer'
                        or release.get('mode') == 'potentiometer_coast')
        required = (required or record_potentiometer
                    or getattr(resolved, 'active_septic_brake_test', False)
                    or getattr(resolved, 'contact_attitude_run', None) is not None)
        if config.get('behavior') == 'level_coast':
            resolved.sense = required
        elif 'sensing' in config:
            if type(config['sensing']) is not bool:
                raise ValueError('Interaction.config.sensing must be boolean')
            if required and not config['sensing']:
                raise ValueError('sensing: false conflicts with potentiometer detection/release, '
                                 'record_potentiometer or sensor-backed experiment')
            resolved.sense = config['sensing']
        else:
            resolved.sense = required or bool(getattr(resolved, 'sense', False))
    validate_interaction_modes(resolved)
    return resolved


def interaction_vicon_diagnostic_args(args, config):
    """Opt-in source-frame diagnostics; no change to tracking/control mode."""
    enabled = (getattr(args, 'vicon_source_metadata', False)
               or config.get('vicon_source_metadata', False) is True)
    return ' --vicon-source-metadata' if enabled else ''


def normalize_active_septic_brake_calibration(args):
    """Expand the fixed-pulse hardware fit entry without changing plain calibration."""
    if (not getattr(args, 'active_septic_brake_calibration', False)
            or getattr(args, '_active_septic_brake_calibration_normalized', False)):
        return
    conflicts = (
        'interaction', 'calibrate', 'braking_test', 'mpc', 'baseline',
        'hover', 'sense', 'vicon_rigidbody_position_only',
        'illumination', 'intractable_illumination', 'morphing',
        'radio', 'ground', 'droneless', 'record', 'skip_confirm',
        'active_septic_brake_test', 'targeted_braking_calibration',
    )
    if (any(getattr(args, name, False) for name in conflicts)
            or getattr(args, 'contact_attitude_run', None) is not None
            or getattr(args, 'adaptive_braking_calibration', None) is not None):
        raise ValueError(
            '--active-septic-brake-calibration is a standalone USB hardware '
            'mode; do not combine it with other flight/sensor/skip-confirm flags'
        )
    args.calibrate = True
    args.vicon_rigidbody_position_only = True
    args.planar_braking_calibration = True
    args._active_septic_brake_calibration_normalized = True


def normalize_active_septic_brake_test(args):
    """Expand the single hardware-test selector without relaxing any gate."""
    if (not getattr(args, 'active_septic_brake_test', False)
            or getattr(args, '_active_septic_brake_test_normalized', False)):
        return
    conflicts = (
        'interaction', 'calibrate', 'braking_test', 'mpc', 'baseline',
        'hover', 'sense', 'vicon_rigidbody_position_only',
        'illumination', 'intractable_illumination', 'morphing',
        'radio', 'ground', 'droneless', 'record', 'skip_confirm',
    )
    if any(getattr(args, name, False) for name in conflicts):
        raise ValueError(
            '--active-septic-brake-test is a standalone USB hardware mode; '
            'do not combine it with other flight/sensor/skip-confirm flags'
        )
    if (getattr(args, 'contact_attitude_run', None) is not None
            or getattr(args, 'adaptive_braking_calibration', None) is not None
            or getattr(args, 'targeted_braking_calibration', False)):
        raise ValueError(
            '--active-septic-brake-test cannot select another experiment'
        )
    args.interaction = True
    args.sense = True
    args.vicon_rigidbody_position_only = True
    args._active_septic_brake_test_normalized = True


def validate_interaction_modes(args, *, defer_sensing=False):
    """Reject ambiguous experiment ownership before opening any connections."""
    normalize_active_septic_brake_calibration(args)
    normalize_active_septic_brake_test(args)
    contact_attitude_run = getattr(args, 'contact_attitude_run', None)
    if (getattr(args, 'vicon_rigidbody_position_only', False)
            and not (getattr(args, 'interaction', False)
                     or getattr(args, 'calibrate', False))):
        raise ValueError(
            '--vicon-rigidbody-position-only requires --interaction or '
            '--calibrate'
        )
    if (getattr(args, 'vicon_rigidbody_position_only', False)
            and contact_attitude_run is not None):
        raise ValueError(
            '--vicon-rigidbody-position-only cannot be combined with '
            '--contact-attitude-run'
        )
    if contact_attitude_run is not None:
        if contact_attitude_run not in (1, 2, 3):
            raise ValueError('--contact-attitude-run must be 1, 2, or 3')
        if not getattr(args, 'interaction', False):
            raise ValueError('--contact-attitude-run requires --interaction')
        if not getattr(args, 'sense', False) and not defer_sensing:
            raise ValueError('--contact-attitude-run requires --sense')
        if getattr(args, 'ground', False) or getattr(args, 'droneless', False):
            raise ValueError('--contact-attitude-run is an airborne hardware protocol')
    if getattr(args, 'targeted_braking_calibration', False) and (
            not getattr(args, 'calibrate', False)
            or getattr(args, 'interaction', False)
            or getattr(args, 'braking_test', False)
            or getattr(args, 'mpc', False)
            or getattr(args, 'ground', False)):
        raise ValueError(
            '--targeted-braking-calibration requires --calibrate without '
            '--interaction, --braking-test, --mpc, or --ground'
        )
    adaptive_braking = getattr(args, 'adaptive_braking_calibration', None)
    if getattr(args, 'planar_braking_calibration', False) and (
            not getattr(args, 'calibrate', False)
            or getattr(args, 'targeted_braking_calibration', False)
            or adaptive_braking is True or getattr(args, 'ground', False)):
        raise ValueError(
            '--planar-braking-calibration requires hardware --calibrate '
            'without targeted/adaptive braking or ground'
        )
    if adaptive_braking is not None and (
            not getattr(args, 'calibrate', False)
            or getattr(args, 'interaction', False)
            or getattr(args, 'braking_test', False)
            or getattr(args, 'mpc', False)):
        raise ValueError(
            '--[no-]adaptive-braking-calibration requires --calibrate '
            'without --interaction, --braking-test, or --mpc'
        )
    modes = (
        'interaction', 'calibrate', 'braking_test', 'mpc', 'baseline', 'hover'
    )
    if sum(bool(getattr(args, mode, False)) for mode in modes) > 1:
        raise ValueError(
            '--interaction, --calibrate, --braking-test, --mpc, --baseline '
            'and --hover are mutually exclusive'
        )
    direction = getattr(args, 'braking_test_direction', None)
    repetitions = getattr(args, 'braking_test_repetitions', None)
    if (direction is not None or repetitions is not None) and not getattr(args, 'braking_test', False):
        raise ValueError('--braking-test-direction and --braking-test-repetitions require --braking-test')
    if direction is not None and direction not in ('both', 'positive-y', 'negative-y'):
        raise ValueError('--braking-test-direction must be both, positive-y, or negative-y')
    if repetitions is not None and (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 1 <= repetitions <= 3
    ):
        raise ValueError('--braking-test-repetitions must be an integer from 1 to 3')
    if getattr(args, 'braking_test', False) and getattr(args, 'sense', False):
        raise ValueError('--braking-test does not use --sense; omit --sense')
    if getattr(args, 'braking_test', False):
        for mode in ('illumination', 'intractable_illumination', 'morphing', 'blender'):
            if getattr(args, mode, False):
                raise ValueError('--braking-test cannot be combined with --' + mode.replace('_', '-'))
    if getattr(args, 'mpc', False):
        if getattr(args, 'sense', False):
            raise ValueError('--mpc does not use --sense; omit --sense')
        if getattr(args, 'ground', False):
            raise ValueError('--mpc is an airborne calibration and cannot use --ground')
        for mode in (
                'illumination', 'intractable_illumination', 'morphing',
                'blender', 'record'):
            if getattr(args, mode, False):
                raise ValueError(
                    '--mpc cannot be combined with --'
                    + mode.replace('_', '-')
                )
    if getattr(args, 'baseline', False):
        if getattr(args, 'sense', False):
            raise ValueError('--baseline does not use --sense; omit --sense')
        if getattr(args, 'ground', False):
            raise ValueError('--baseline is airborne and cannot use --ground')
        for mode in (
                'illumination', 'intractable_illumination', 'morphing',
                'blender', 'record', 'droneless'):
            if getattr(args, mode, False):
                raise ValueError(
                    '--baseline cannot be combined with --'
                    + mode.replace('_', '-')
                )
    if getattr(args, 'hover', False):
        if getattr(args, 'sense', False):
            raise ValueError('--hover does not use --sense; omit --sense')
        if getattr(args, 'ground', False):
            raise ValueError('--hover is airborne and cannot use --ground')
        for mode in (
                'illumination', 'intractable_illumination', 'morphing',
                'blender', 'record', 'droneless'):
            if getattr(args, mode, False):
                raise ValueError(
                    '--hover cannot be combined with --'
                    + mode.replace('_', '-')
                )


def interaction_application_flag(args):
    """Select the single controller experiment mode (legacy default is interaction)."""
    validate_interaction_modes(args)
    if getattr(args, 'hover', False):
        return '--hover'
    if getattr(args, 'baseline', False):
        return '--baseline'
    if getattr(args, 'mpc', False):
        return '--mpc'
    if getattr(args, 'braking_test', False):
        return '--braking-test'
    if getattr(args, 'calibrate', False):
        return '--calibrate'
    return '--interaction'


def interaction_braking_test_args(args):
    """Forward explicit repeat-test selectors without changing legacy commands."""
    validate_interaction_modes(args)
    options = []
    for name in ('braking_test_direction', 'braking_test_repetitions'):
        value = getattr(args, name, None)
        if value is not None:
            options.extend(('--' + name.replace('_', '-'), str(value)))
    return (' ' + ' '.join(options)) if options else ''


def interaction_adaptive_calibration_args(args):
    """Keep plain calibration XYZ-only; forward explicit planar selections."""
    validate_interaction_modes(args)
    if not getattr(args, 'calibrate', False):
        return ''
    if (getattr(args, 'targeted_braking_calibration', False)
            or getattr(args, 'adaptive_braking_calibration', None) is False):
        return ' --no-adaptive-braking-calibration'
    if getattr(args, 'adaptive_braking_calibration', None) is True:
        return ' --adaptive-braking-calibration'
    # Be explicit so an older deployed controller whose local default was
    # adaptive cannot silently restore the retired second stage.
    return ' --no-adaptive-braking-calibration'


def interaction_targeted_calibration_args(args):
    """Forward the deterministic high-speed calibration only when opted in."""
    validate_interaction_modes(args)
    if getattr(args, 'targeted_braking_calibration', False):
        return ' --targeted-braking-calibration'
    return ''


def interaction_contact_attitude_args(args):
    """Forward the validated three-flight selector to the drone controller."""
    validate_interaction_modes(args)
    run = getattr(args, 'contact_attitude_run', None)
    return '' if run is None else f' --contact-attitude-run {run}'


def contact_attitude_skips_pointcloud_dispatcher(args):
    """Rigid-body comparison runs already have stable manifest identities."""
    return (getattr(args, 'contact_attitude_run', None) in (2, 3)
            or getattr(args, 'vicon_rigidbody_position_only', False))


def interaction_mocap_args(
        drone, wrench_mode=False, contact_attitude_run=None,
        rigidbody_position_only=False):
    """Build interaction mocap arguments using the pre-wrench defaults.

    ``wrench_mode`` remains in the signature for caller compatibility. Onboard
    wrench detection does not require mocap orientation, so a drone with an
    initial marker position defaults to pointcloud localization as before.
    """
    position = drone.get('init_pos')
    object_name = drone.get('obj_name', drone.get('label'))
    if rigidbody_position_only:
        if contact_attitude_run is not None:
            raise ValueError(
                'rigid-body position-only route cannot select a contact '
                'attitude run'
            )
        if not object_name:
            raise ValueError(
                f"Drone {drone['id']} requires obj_name or label for "
                'rigid-body position-only Vicon'
            )
        return f"--vicon-rigidbody-position-only {shlex.quote(str(object_name))} "
    if contact_attitude_run is not None:
        if contact_attitude_run not in (1, 2, 3):
            raise ValueError('contact_attitude_run must be 1, 2, or 3')
        if contact_attitude_run == 1:
            if not position:
                raise ValueError(
                    f"Drone {drone['id']} requires init_pos for contact attitude run 1"
                )
            return (
                f"--init-pos {position[0]} {position[1]} {position[2]} "
                "--vicon-mode pointcloud "
            )
        if not object_name:
            raise ValueError(
                f"Drone {drone['id']} requires obj_name or label for "
                f"contact attitude run {contact_attitude_run}"
            )
        full_pose = ' --vicon-full-pose' if contact_attitude_run == 3 else ''
        return (
            f"--obj-name {object_name} --vicon-mode rigidbody{full_pose} "
        )
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
