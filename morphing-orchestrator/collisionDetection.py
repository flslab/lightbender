import yaml
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import copy
import argparse
import sys


# -----------------------------
# Data Models
# -----------------------------

@dataclass
class LEDConfig:
    mode: str
    rate: float
    formula: Optional[str] = None


@dataclass
class DroneParams:
    linear: bool
    relative: bool


@dataclass
class Drone:
    name: str
    target: List[float]
    waypoints: List[List[float]]
    delta_t: float
    iterations: int
    params: DroneParams
    servos: List[List[float]]
    pointers: List[List[float]]
    led: LEDConfig


@dataclass
class Scene:
    cameraLocation: List[int]
    drones: Dict[str, Drone] = field(default_factory=dict)

# 1. Define this custom dumper at the top of your file to prevent `&id001` alias tags
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

class Replacement:
    SAFE_SPEED = 300.0  # mm/sec (tune for your drones)
    SAFETY_DELAY = 1.0  # seconds buffer to prevent overlap

    def __init__(self, args):
        self.args = args
        self.distance = args.distance
        self.mission_file = args.mission
        self.mission = self.read_in_mission_file()
        self.input_scene = self.parse_in_mission_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    # -----------------------------
    # File Handling
    # -----------------------------

    def read_in_mission_file(self):
        print(self.mission_file)
        with open(self.mission_file, 'r') as f:
            return yaml.safe_load(f)

    def parse_in_mission_file(self) -> Scene:
        drones = {}

        for name, drone_data in self.mission["drones"].items():
            if drone_data is None:
                raise ValueError(f"Drone '{name}' has no configuration in mission.yaml")
            
            drone = Drone(
                name=name,
                target=drone_data["target"],
                waypoints=drone_data["waypoints"],
                delta_t=drone_data["delta_t"],
                iterations=drone_data["iterations"],
                params=DroneParams(**drone_data["params"]),
                servos=drone_data["servos"],
                pointers=drone_data["pointers"],
                led=LEDConfig(**drone_data["led"]),
            )
            drones[name] = drone

        return Scene(drones=drones, cameraLocation=[2.35, 0.0, 0.88])
    

    # convert current waypoints into absolute time instead of duration to reach specific waypoint
    def convert_to_absolute_time(self, waypoints):
        absolute_time = 0.0
        new_waypoints = []

        for i, wp in enumerate(waypoints):
            x, y, z, yaw, duration = wp

            if i == 0:
                absolute_time = 0.0
            else:
                absolute_time += duration

            new_waypoints.append([x, y, z, yaw, absolute_time])

        return new_waypoints
    
    def interpolate_position(self, waypoints, t):
        waypoints = self.convert_to_absolute_time(waypoints)

        # Before mission start
        if t <= waypoints[0][4]:
            return waypoints[0][:3]

        # After mission end
        if t >= waypoints[-1][4]:
            return waypoints[-1][:3]

        # Find segment containing t
        for i in range(len(waypoints) - 1):
            t0 = waypoints[i][4]
            t1 = waypoints[i + 1][4]

            if t0 <= t <= t1:
                x0, y0, z0 = waypoints[i][:3]
                x1, y1, z1 = waypoints[i + 1][:3]

                # Hover case (no time difference)
                if t1 == t0:
                    return [x0, y0, z0]

                alpha = (t - t0) / (t1 - t0)

                # Linear interpolation, assumes a straight line when UAV travels
                x = x0 + alpha * (x1 - x0)
                y = y0 + alpha * (y1 - y0)
                z = z0 + alpha * (z1 - z0)

                return [x, y, z]
            
    def distance_3d(self, p1, p2):
        return math.sqrt(
            (p2[0] - p1[0]) ** 2 +
            (p2[1] - p1[1]) ** 2 +
            (p2[2] - p1[2]) ** 2
        )
    
    def check_collision(self, waypoints_E, waypoints_F, delta, time_step=0.05):
        abs_E = self.convert_to_absolute_time(waypoints_E)
        abs_F = self.convert_to_absolute_time(waypoints_F)

        start_time = 0.0
        end_time = max(abs_E[-1][4], abs_F[-1][4])

        t = start_time

        while t <= end_time:

            pos_E = self.interpolate_position(waypoints_E, t)
            pos_F = self.interpolate_position(waypoints_F, t)

            dist = self.distance_3d(pos_E, pos_F)

            if dist <= delta:
                return {
                    "time": t,
                    "E_position": pos_E,
                    "F_position": pos_F,
                    "distance": dist
                }

            t += time_step

        return None
    
    def collision_detection(self):
        waypoints_E = self.input_scene.drones["lb3"].waypoints
        waypoints_F = self.input_scene.drones["lb2"].waypoints

        collision = self.check_collision(waypoints_E, waypoints_F, self.distance)

        if collision:
            print("Collision detected!")
            print(f"Time: {collision['time']:.4f} seconds")
            print(f"FLS E Position: {collision['E_position']}")
            print(f"FLS F Position: {collision['F_position']}")
            print(f"Distance: {collision['distance']:.6f}")
        else:
            print("No collision detected.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mission", type=str)
    parser.add_argument("--distance", type=float)

    args = parser.parse_args()

    with Replacement(args) as r:
        r.collision_detection()