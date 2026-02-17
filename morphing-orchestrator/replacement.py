
def empty_battery_replacement_position(xcam, ycam, zcam, xe, ye, ze):
    # vector d = p_EF - p_cam
    dx = xe - xcam
    dy = ye - ycam
    dz = ze - zcam

    # magnitude |d|
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    if dist == 0.0:
        raise ValueError("Camera and EF drone occupy the same position; unit vector undefined.")

    # unit vector hat(d)
    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    # offset magnitude = 25 cm = 0.25 m
    offset_m = 0.25

    # choose p_R = p_EF - 0.25 * hat(d)  (i.e., 25 cm toward the camera)
    xr = xe - offset_m * ux
    yr = ye - offset_m * uy
    zr = ze - offset_m * uz


    return [xr,yr,zr]

def create_manifest():


"""
How this will work:
1. FLS sends low battery message to orchestrator with current waypoint
2. Orchestrator creates two mission files:
    1. mission file using E's current coordinate for replacing E
    2. mission file for E to escape to
3. Orchestrator sends both mission files to each drone along with E's original mission
"""