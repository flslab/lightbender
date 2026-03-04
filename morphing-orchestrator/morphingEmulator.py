import yaml
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import copy


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
    drones: Dict[str, Drone] = field(default_factory=dict)

# 1. Define this custom dumper at the top of your file to prevent `&id001` alias tags
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

# -----------------------------
# Replacement Logic
# -----------------------------

class Replacement:
    SAFE_SPEED = 300.0  # mm/sec (tune for your drones)
    SAFETY_DELAY = 1.0  # seconds buffer to prevent overlap

    def __init__(self, args):
        self.args = args
        self.mission_file = "debug.yaml"
        self.mission = self.read_in_mission_file(self.mission_file)
        self.input_scene = self.parse_in_mission_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    # -----------------------------
    # File Handling
    # -----------------------------

    def read_in_mission_file(self, mission_file):
        with open(mission_file, 'r') as f:
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

        return Scene(drones=drones)

    def generate_replacement_pose(self, empty_fls, full_fls):
        coordinate_E = empty_fls.target
        coordinate_F = full_fls.target

        # Compute x coordinate
        adjacent = coordinate_E[0] + 2.35
        alpha = math.atan(1.56/adjacent)
        beta = 0.25
        epsilon = math.tan(math.pi/2 - alpha)*beta

        replacement_x = coordinate_E[0] - 1.56 + epsilon
        replacement_y = coordinate_E[1] - beta
        replacement_z = coordinate_E[2]

        return [replacement_x, replacement_y, replacement_z, coordinate_E[3], 3]


    def replace_pose_data(self):
        replacement_drone_waypoints = []

        empty_fls = self.input_scene.drones["lb2"]
        full_fls = self.input_scene.drones["lb3"]

        replacement_drone_origin_waypoint = full_fls.target

        replacement_drone_enter_scene = self.generate_replacement_pose(empty_fls, full_fls)
        
        replacement_drone_first_waypoint = replacement_drone_origin_waypoint
        replacement_drone_first_waypoint[1] = replacement_drone_enter_scene[1] 
        replacement_drone_first_waypoint[0] = replacement_drone_enter_scene[0] - 0.25

        replacement_drone_waypoints.append(replacement_drone_origin_waypoint)
        replacement_drone_waypoints.append(replacement_drone_first_waypoint)
        replacement_drone_waypoints.append(replacement_drone_enter_scene)
        replacement_drone_waypoints.append(full_fls.target)

        return replacement_drone_waypoints

    def generate_output_scene(self):
        replacement_waypoints = self.replace_pose_data()
        self.input_scene.drones["lb3"].waypoints = replacement_waypoints + self.input_scene.drones["lb2"].waypoints
        self.input_scene.drones["lb2"].waypoints += replacement_waypoints

        output_yaml = {
            "name": "Scene",
            "drones": {}
        }

        for name, drone in self.input_scene.drones.items():
            output_yaml["drones"][name] = {
                "target": drone.target,
                "waypoints": drone.waypoints,
                "delta_t": drone.delta_t,
                "iterations": drone.iterations,
                "params": {
                    "linear": drone.params.linear,
                    "relative": drone.params.relative
                },
                "servos": drone.servos,
                "pointers": drone.pointers,
                "led": {
                    "mode": drone.led.mode,
                    "rate": drone.led.rate,
                    "formula": drone.led.formula
                }
            }

        print(output_yaml)

        with open("mission.yaml", "w") as f:
            yaml.dump(output_yaml, f, sort_keys=False, Dumper=NoAliasDumper, default_flow_style=None)

        print("Swap mission generated (matching mission.YAML structure).")


# -----------------------------
# Run
# -----------------------------

if __name__ == '__main__':
    args = None
    with Replacement(args) as r:
        r.generate_output_scene()