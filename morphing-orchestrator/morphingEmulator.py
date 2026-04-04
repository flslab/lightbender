import yaml
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import copy
import argparse


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
    SAFE_SPEED = 300.0
    SAFETY_DELAY = 1.0

    def __init__(self, args):
        self.args = args
        self.mission_file = args.mission
        self.morphing_technique = args.morphing
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

        return Scene(drones=drones, cameraLocation=[2.2769, -0.0756, 0.9199])

    def morphing_technique_1_replacement_pose(self, empty_fls):
        coordinate_E = empty_fls.waypoints[0]

        replacement_x = coordinate_E[0] - 0.3
        replacement_y = coordinate_E[1]
        replacement_z = coordinate_E[2]

        return [replacement_x, replacement_y, replacement_z, 0.0, 3.0]
    
    def morphing_technique_2_replacement_pose(self, empty_fls):

        ### Find the replacement waypoint's position (x,y,z)
        coordinate_E = empty_fls.waypoints[0]

        # Compute y coordinate
        adjacent = math.fabs(coordinate_E[0] - self.input_scene.cameraLocation[0])
        alpha = math.atan(1.56/adjacent)
        beta = 0.5
        epsilon = math.atan(alpha)*beta

        replacement_x = coordinate_E[0] - beta
        replacement_y = coordinate_E[1] + epsilon
        replacement_z = coordinate_E[2]

        return [replacement_x, replacement_y, replacement_z, coordinate_E[3], 3.0]

    def replace_pose_data_full_fls(self):
        replacement_drone_waypoints = []

        empty_fls = self.input_scene.drones["lb3"]
        full_fls = self.input_scene.drones["lb2"]

        replacement_drone_origin_waypoint = full_fls.target

        if(self.morphing_technique == "1"):
            replacement_drone_enter_scene = self.morphing_technique_1_replacement_pose(empty_fls)

        elif(self.morphing_technique == "2"):
            replacement_drone_enter_scene = self.morphing_technique_2_replacement_pose(empty_fls)
        
        replacement_drone_first_waypoint = replacement_drone_origin_waypoint
        replacement_drone_first_waypoint[0] = replacement_drone_enter_scene[0]

        for i in full_fls.waypoints:
            replacement_drone_waypoints.append(i)

        replacement_drone_waypoints.append(replacement_drone_first_waypoint)
        replacement_drone_waypoints.append(replacement_drone_enter_scene)
        replacement_drone_waypoints.append(empty_fls.target)
        replacement_drone_waypoints.append(empty_fls.target)

        return replacement_drone_waypoints
    
    def replace_pose_data_empty_fls(self):
        # Fail on waypoint 2
        waypoint_failure = 2

        replacement_drone_waypoints = copy.deepcopy(self.input_scene.drones["lb3"].waypoints[:waypoint_failure])

        empty_fls = self.input_scene.drones["lb3"]
        full_fls = self.input_scene.drones["lb2"]

        #Stay at current waypoint until replacement fls is in position, append twice since this is what is needed for full fls to be in position
        replacement_drone_waypoints.append(replacement_drone_waypoints[waypoint_failure-1])
        replacement_drone_waypoints.append(replacement_drone_waypoints[waypoint_failure-1])


        #Send empty drone away
        fly_out_waypoint = empty_fls.waypoints[0]
        fly_out_waypoint[1] = full_fls.waypoints[0][1]

        replacement_drone_waypoints.append(fly_out_waypoint)
        replacement_drone_waypoints.append(full_fls.waypoints[0])

        return replacement_drone_waypoints

    def replace_waypoints(self):
        #work with deep copies
        self.input_scene.drones["lb2"].waypoints = self.replace_pose_data_full_fls()
        self.input_scene.drones["lb3"].waypoints = self.replace_pose_data_empty_fls()

        original_lb2_waypoints = copy.deepcopy(self.input_scene.drones["lb2"].waypoints)
        original_lb3_waypoints = copy.deepcopy(self.input_scene.drones["lb3"].waypoints)

        self.input_scene.drones["lb2"].waypoints += original_lb3_waypoints
        self.input_scene.drones["lb3"].waypoints += original_lb2_waypoints

        self.input_scene.drones["lb2"].target = self.input_scene.drones["lb2"].waypoints[0]

        # put a safe value of 5 seconds for time duration of each waypoint
        for drone in self.input_scene.drones.values():
            for waypoint in drone.waypoints:
                if len(waypoint) > 0:
                    waypoint[-1] = 2.0


    def generate_lighting(self):
        # Use the empty FLS first waypoint to define the same reference geometry
        empty_fls = self.input_scene.drones["lb3"]
        empty_fls_x = empty_fls.waypoints[0][0]
        empty_fls_y = empty_fls.waypoints[0][1]
        
        # Used for formula replacement of the LED lighting
        right_boundary_slope = (self.input_scene.cameraLocation[1] - (empty_fls_y + 0.156))/(self.input_scene.cameraLocation[0] - empty_fls_x)
        right_boundary_intercept = self.input_scene.cameraLocation[1] - right_boundary_slope*self.input_scene.cameraLocation[0]
        left_boundary_slope = (self.input_scene.cameraLocation[1] - (empty_fls_y - 0.156))/(self.input_scene.cameraLocation[0] - empty_fls_x)
        left_boundary_intercept = self.input_scene.cameraLocation[1] - left_boundary_slope*self.input_scene.cameraLocation[0]

        visible_formula = (
            f"[255, 0, 0] if ({left_boundary_slope} * (x + 0.0797) + {left_boundary_intercept}) < (y - 0.156 + 0.00624*i) < ({right_boundary_slope} * (x + 0.0797) + {right_boundary_intercept}) else [0, 0, 0]"
        )

        # Apply the same lighting rule to both drones.
        for name, drone in self.input_scene.drones.items():
            drone.led.mode = "expression"
            drone.led.rate = 50.0
            drone.led.formula = visible_formula

    def generate_output_scene(self):
        self.replace_waypoints()
        self.generate_lighting()

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
                "led": {
                    "mode": drone.led.mode,
                    "rate": drone.led.rate,
                    "formula": drone.led.formula
                }
            }

        with open("mission.yaml", "w") as f:
            yaml.dump(output_yaml, f, sort_keys=False, Dumper=NoAliasDumper, default_flow_style=None)

        print("Swap mission generated (matching mission.YAML structure).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mission", type=str)
    parser.add_argument("--morphing", type=str)

    args = parser.parse_args()

    with Replacement(args) as r:
        r.generate_output_scene()