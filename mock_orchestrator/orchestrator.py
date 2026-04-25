import argparse
import json
import socket
import threading
import time
from pathlib import Path

import yaml


DEFAULT_MANIFEST = Path(__file__).with_name("swarm_manifest.yaml")


class BlenderMonitorClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.confirm_launch_event = threading.Event()
        self.remote_stop_event = threading.Event()
        self._lock = threading.Lock()

    def connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self.host, self.port))
        sock.settimeout(1.0)
        self.sock = sock

        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def close(self):
        self.stop_event.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def send(self, payload):
        if self.sock is None:
            raise RuntimeError("Not connected to Blender monitor.")
        data = (json.dumps(payload) + "\n").encode("utf-8")
        with self._lock:
            self.sock.sendall(data)

    def _reader_loop(self):
        buffer = ""
        while not self.stop_event.is_set():
            try:
                data = self.sock.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._handle_incoming(msg)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_incoming(self, msg):
        cmd = msg.get("cmd")
        if cmd == "confirm_launch":
            self.confirm_launch_event.set()
            print("\n[Blender -> Mock] confirm_launch")
        elif cmd == "stop":
            self.remote_stop_event.set()
            print("\n[Blender -> Mock] stop")
        else:
            print(f"\n[Blender -> Mock] {msg}")


def load_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_mission_path(manifest, manifest_path: Path):
    controller = manifest.get("controller", {})
    mission_file = controller.get("mission_file")
    if not mission_file:
        return None
    mission_root = (manifest_path.parent / controller.get("mission_path", "SFL")).resolve()
    return mission_root / mission_file


def resolve_mission_drone_ids(manifest, manifest_path: Path):
    mission_path = resolve_mission_path(manifest, manifest_path)
    if mission_path is None or not mission_path.exists():
        return None
    with mission_path.open("r", encoding="utf-8") as handle:
        mission = yaml.safe_load(handle) or {}
    drones = mission.get("drones", {})
    if isinstance(drones, dict):
        return set(drones.keys())
    return None


def build_drone_payloads(manifest, manifest_path: Path, selected_ids=None):
    mission_path = resolve_mission_path(manifest, manifest_path)
    mission_drones = {}
    if mission_path and mission_path.exists():
        with mission_path.open("r", encoding="utf-8") as handle:
            mission = yaml.safe_load(handle) or {}
            mission_drones = mission.get("drones", {})

    mission_ids = set(mission_drones.keys()) if mission_drones else None
    if selected_ids:
        mission_ids = set(selected_ids)

    default_height = manifest.get('common', {}).get('default_height',
                     manifest.get('mission', {}).get('default_height', 0.24))

    drones = []
    for drone in manifest.get("drones", []):
        drone_id = drone.get("id")
        if mission_ids and drone_id not in mission_ids:
            continue
            
        drone_mission = mission_drones.get(drone_id) if isinstance(mission_drones, dict) else None
        if drone_mission and 'target' in drone_mission:
            target = drone_mission['target']
            init_pos = [target[0], target[1], default_height]
        else:
            init_pos = drone.get("init_pos", [0, 0, 0])

        drones.append(
            {
                "id": drone_id,
                "ip": drone.get("ip", ""),
                "init_pos": init_pos,
                "status": "idle",
            }
        )
    return drones


def prompt_step(step_name: str, description: str, messages):
    while True:
        print(f"\n=== {step_name} ===")
        print(description)
        print(f"Messages in this group: {len(messages)}")
        cmd = input("Enter=send, p=preview, q=quit: ").strip().lower()
        if cmd == "":
            return True
        if cmd == "p":
            for idx, message in enumerate(messages, start=1):
                print(f"[{idx}] {json.dumps(message)}")
            continue
        if cmd == "q":
            return False
        print("Unknown command. Use Enter, p, or q.")


def wait_for_launch_confirmation(client: BlenderMonitorClient):
    while True:
        if client.remote_stop_event.is_set():
            return False
        if client.confirm_launch_event.is_set():
            return True
        cmd = input(
            "\nLaunch gate: w=wait for Blender confirm, b=bypass locally, q=quit: "
        ).strip().lower()
        if cmd == "w":
            print("Waiting for Blender to send confirm_launch...")
            while not client.confirm_launch_event.wait(timeout=0.25):
                if client.remote_stop_event.is_set():
                    return False
            return True
        if cmd == "b":
            print("Bypassing Blender confirmation for this mock run.")
            return True
        if cmd == "q":
            return False
        print("Unknown command. Use w, b, or q.")


def send_group(client: BlenderMonitorClient, name: str, messages, delay: float):
    print(f"Sending group '{name}'...")
    for message in messages:
        if client.remote_stop_event.is_set():
            print("Stop requested by Blender. Halting group send.")
            return False
        client.send(message)
        print(f"[Mock -> Blender] {json.dumps(message)}")
        if delay > 0:
            time.sleep(delay)
    return True


def build_sequence(drones, battery_threshold):
    drone_ids = [d["id"] for d in drones]
    ready_messages = []
    landed_messages = []

    for index, drone in enumerate(drones):
        ready_messages.append(
            {
                "cmd": "drone_status",
                "id": drone["id"],
                "status": "ready",
                "battery": round(8.18 - (index * 0.03), 2),
            }
        )
        landed_messages.append(
            {
                "cmd": "drone_status",
                "id": drone["id"],
                "status": "landed",
                "battery": round(8.05 - (index * 0.04), 2),
            }
        )

    return [
        {
            "name": "swarm_info",
            "description": f"Advertise {len(drone_ids)} drones to the Blender add-on.",
            "messages": [
                {
                    "cmd": "swarm_info",
                    "battery_threshold": battery_threshold,
                    "drones": drones,
                }
            ],
        },
        {
            "name": "booting",
            "description": "Mark all drones as booting.",
            "messages": [
                {"cmd": "drone_status", "id": drone_id, "status": "booting"}
                for drone_id in drone_ids
            ],
        },
        {
            "name": "ready",
            "description": "Mark all drones as ready so Blender can enable launch confirmation.",
            "messages": ready_messages,
            "wait_for_launch": True,
        },
        {
            "name": "landed",
            "description": "Simulate the illumination ending and the drones landing.",
            "messages": landed_messages,
        },
        {
            "name": "cleanup",
            "description": "Notify Blender that logs were fetched and the mock orchestrator is done.",
            "messages": [
                {"cmd": "logs_fetched"},
                {"cmd": "all_stopped"},
            ],
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mock LightBender orchestrator that talks to the Blender add-on."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Blender monitor host.")
    parser.add_argument("--port", type=int, default=5598, help="Blender monitor port.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to swarm manifest YAML.",
    )
    parser.add_argument(
        "--drone-id",
        action="append",
        default=[],
        help="Optional drone ID filter. Repeat to include multiple drones.",
    )
    parser.add_argument(
        "--message-delay",
        type=float,
        default=0.15,
        help="Delay in seconds between messages inside a group.",
    )
    parser.add_argument(
        "--auto-cleanup-on-stop",
        action="store_true",
        help="Send logs_fetched and all_stopped automatically if Blender sends stop.",
    )
    # Dummy args to avoid crashing when Blender calls orchestrator.py
    parser.add_argument("-l", "--illumination", action="store_true")
    parser.add_argument("--interaction", action="store_true")
    parser.add_argument("--intractable-illumination", action="store_true")
    parser.add_argument("--dark", action="store_true")
    parser.add_argument("--blender", type=str, default="", help="Connect to Blender monitor via IP:PORT")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Process --blender argument
    host, port = args.host, args.port
    if args.blender:
        parts = args.blender.split(':')
        if len(parts) == 2:
            host = parts[0]
            port = int(parts[1])
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    battery_threshold = manifest.get("mission", {}).get("battery_threshold", 7)
    drones = build_drone_payloads(manifest, manifest_path, selected_ids=args.drone_id)

    if not drones:
        raise SystemExit("No drones selected from the manifest/mission.")

    print("Mock orchestrator configuration:")
    print(f"  manifest: {manifest_path}")
    print(f"  monitor : {host}:{port}")
    print(f"  drones  : {', '.join(d['id'] for d in drones)}")
    print("")
    print(f"Open the Blender swarm monitor first so it starts listening on port {port}.")

    client = BlenderMonitorClient(host, port)
    try:
        client.connect()
        print("Connected to Blender monitor.")
        sequence = build_sequence(drones, battery_threshold)

        for step in sequence:
            if client.remote_stop_event.is_set():
                print("Stop requested before next step.")
                break

            should_send = prompt_step(step["name"], step["description"], step["messages"])
            if not should_send:
                print("User aborted mock run.")
                break

            if not send_group(client, step["name"], step["messages"], args.message_delay):
                break

            if step.get("wait_for_launch"):
                if not wait_for_launch_confirmation(client):
                    print("Launch phase aborted.")
                    break

        if client.remote_stop_event.is_set() and args.auto_cleanup_on_stop:
            cleanup = [{"cmd": "logs_fetched"}, {"cmd": "all_stopped"}]
            send_group(client, "cleanup_on_stop", cleanup, args.message_delay)
    finally:
        client.close()
        print("Mock orchestrator disconnected.")


if __name__ == "__main__":
    main()
