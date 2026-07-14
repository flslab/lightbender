#!/usr/bin/env python3
"""
Standalone SDP31 sensor node — same pattern as camera_node.

Reads swarm_manifest.yaml for controller ZMQ address (orchestrator pushes it each run).
"""

import argparse
import json
import logging
import os
import sys
import threading
import time

MANIFEST_FILE = "swarm_manifest.yaml"
OUTPUT_FILENAME = "sensor_log.json"
NODE_ID = "SENSOR"
STATUS_LOG_SAVED = "LOG_SAVED"
STATUS_LOG_SAVE_FAILED = "LOG_SAVE_FAILED"

SDP31_I2C_ADDR = 0x21
CMD_SOFT_RESET = 0x0006
CMD_START_CONTINUOUS = 0x3603
CMD_STOP = 0x3FE9
PRESSURE_SCALE = 60.0
TEMPERATURE_SCALE = 200.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Sensor")


def _sensirion_crc(data):
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _to_int16(msb, lsb):
    raw = (msb << 8) | lsb
    if raw >= 0x8000:
        raw -= 0x10000
    return raw


def _check_crc_triplet(data, label):
    if _sensirion_crc(data[0:2]) != data[2]:
        raise RuntimeError(f"CRC mismatch on {label}")


class SDP31:
    def __init__(self, bus_num=1, address=SDP31_I2C_ADDR):
        try:
            import smbus2
        except ImportError as exc:
            raise SystemExit("Install smbus2: pip install smbus2") from exc

        self._smbus2 = smbus2
        self._bus = smbus2.SMBus(bus_num)
        self._address = address

    def _write_command(self, command):
        msg = self._smbus2.i2c_msg.write(self._address, [
            (command >> 8) & 0xFF,
            command & 0xFF,
        ])
        self._bus.i2c_rdwr(msg)
        time.sleep(0.01)

    def reset(self):
        self._write_command(CMD_SOFT_RESET)
        time.sleep(0.05)

    def start(self):
        self._write_command(CMD_START_CONTINUOUS)
        time.sleep(0.08)

    def stop(self):
        self._write_command(CMD_STOP)

    def read(self):
        read_msg = self._smbus2.i2c_msg.read(self._address, 9)
        self._bus.i2c_rdwr(read_msg)
        data = list(read_msg)

        _check_crc_triplet(data[0:3], "differential pressure")
        _check_crc_triplet(data[3:6], "temperature")
        _check_crc_triplet(data[6:9], "scale factor")

        dp_ticks = _to_int16(data[0], data[1])
        temp_ticks = _to_int16(data[3], data[4])
        scale_factor = _to_int16(data[6], data[7])
        if scale_factor == 0:
            scale_factor = PRESSURE_SCALE

        return dp_ticks / scale_factor, temp_ticks / TEMPERATURE_SCALE

    def ensure_continuous(self):
        """Start continuous measurement; tolerate sensor already running."""
        try:
            self.reset()
        except OSError as exc:
            logger.warning("SDP31 soft reset skipped: %s", exc)
        try:
            self.start()
        except OSError as exc:
            logger.warning("SDP31 start skipped (may already be running): %s", exc)

    def close(self):
        try:
            self.stop()
        except OSError:
            pass
        self._bus.close()


def load_manifest():
    import yaml
    with open(MANIFEST_FILE, "r") as f:
        return yaml.safe_load(f)


class SensorNode:
    def __init__(self, manifest, args, standalone=False):
        self.manifest = manifest
        self.args = args
        self.logging_active = False
        self.samples = []
        self.start_time = None
        self._sensor = None
        self._log_thread = None
        self.context = None
        self.sub_socket = None
        self.push_socket = None

        if not standalone:
            import zmq

            self.ctrl = manifest["controller"]
            self.context = zmq.Context()
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.connect(f"tcp://{self.ctrl['ip']}:{self.ctrl['zmq_cmd_port']}")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

            self.push_socket = self.context.socket(zmq.PUSH)
            self.push_socket.connect(f"tcp://{self.ctrl['ip']}:{self.ctrl['zmq_ack_port']}")

    def start_logging(self):
        if self.logging_active:
            logger.info("Already logging.")
            return

        self.samples = []
        self.start_time = time.time()
        self._sensor = SDP31(bus_num=self.args.bus, address=self.args.address)
        self._sensor.ensure_continuous()
        self.logging_active = True
        self._log_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._log_thread.start()
        logger.info(
            "Logging at %.1f Hz (bus %d, addr 0x%02x)",
            self.args.hz, self.args.bus, self.args.address,
        )

    def _sample_loop(self):
        interval = 1.0 / self.args.hz
        while self.logging_active:
            loop_start = time.time()
            try:
                dp_pa, temp_c = self._sensor.read()
                self.samples.append({"t": loop_start, "dp_pa": dp_pa, "temp_c": temp_c})
            except OSError as exc:
                logger.error("I2C read error: %s", exc)
            sleep_for = interval - (time.time() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _notify_log_saved(self, samples, error=None):
        if not self.push_socket:
            return
        status = STATUS_LOG_SAVE_FAILED if error else STATUS_LOG_SAVED
        msg = {
            "id": NODE_ID,
            "status": status,
            "samples": samples,
            "file": OUTPUT_FILENAME,
        }
        if error:
            msg["error"] = error
        self.push_socket.send_json(msg)
        if error:
            logger.error("Notified orchestrator: log save failed (%s)", error)
        else:
            logger.info("Notified orchestrator: log saved (%d samples)", samples)

    def stop_logging(self):
        if self.logging_active:
            self.logging_active = False
            if self._log_thread:
                self._log_thread.join(timeout=5)
                self._log_thread = None
            if self._sensor:
                try:
                    self._sensor.close()
                except OSError:
                    pass
                self._sensor = None

        if not self.samples and self.start_time is None:
            logger.error("No active logging to stop.")
            self._notify_log_saved(0, error="no active logging session")
            return False

        payload = {
            "tag": self.args.tag,
            "init_pos": self.args.init_pos,
            "start_time": self.start_time,
            "stop_time": time.time(),
            "sample_hz": self.args.hz,
            "i2c_bus": self.args.bus,
            "i2c_address": self.args.address,
            "samples": self.samples,
        }
        tmp_path = OUTPUT_FILENAME + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, OUTPUT_FILENAME)
            logger.info("Saved %d samples to %s", len(self.samples), OUTPUT_FILENAME)
            self._notify_log_saved(len(self.samples))
            return True
        except OSError as exc:
            logger.error("Failed to save sensor log: %s", exc)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            self._notify_log_saved(len(self.samples), error=str(exc))
            return False

    def run(self):
        logger.info("Node online. Waiting for orchestrator commands.")
        self.push_socket.send_json({"id": NODE_ID, "status": "READY"})

        try:
            while True:
                msg = self.sub_socket.recv_json()
                cmd = msg.get("cmd")

                if cmd == "START":
                    self.start_logging()
                elif cmd in ("STOP_SENSOR", "EMERGENCY", "SHUTDOWN"):
                    if cmd == "EMERGENCY":
                        logger.warning("Emergency received.")
                    self.stop_logging()
                    break
        except KeyboardInterrupt:
            self.stop_logging()
        except Exception as exc:
            logger.error("Error: %s", exc)
            self.stop_logging()

    def run_standalone(self):
        logger.info("Standalone mode — logging until Ctrl+C or --duration.")
        try:
            self.start_logging()
            if self.args.duration:
                time.sleep(self.args.duration)
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopped.")
        except Exception as exc:
            logger.error("Error: %s", exc)
        finally:
            self.stop_logging()


def main():
    parser = argparse.ArgumentParser(description="SDP31 sensor node for swarm orchestrator")
    parser.add_argument("--tag", default=None, help="Mission tag for log metadata")
    parser.add_argument("--init-pos", type=float, nargs=3, default=None,
                        help="Vicon marker on sensor (reference) x y z")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", type=lambda x: int(x, 0), default=SDP31_I2C_ADDR)
    parser.add_argument("--standalone", action="store_true",
                        help="log locally without orchestrator (no ZMQ, no manifest)")
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds to log in standalone mode (default: until Ctrl+C)")
    args = parser.parse_args()

    if args.hz <= 0:
        raise SystemExit("--hz must be > 0")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be > 0")

    if args.standalone:
        node = SensorNode({}, args, standalone=True)
        node.run_standalone()
    else:
        SensorNode(load_manifest(), args).run()


if __name__ == "__main__":
    main()
