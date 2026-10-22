#!/usr/bin/env python3
"""Continuously inspect an ADS1115 connected to Raspberry Pi I2C bus 1."""

import argparse
import collections
import statistics
import sys
import time

import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.ads1x15 import Pin
from adafruit_ads1x15.analog_in import AnalogIn


FULL_SCALE_VOLTS = {
    2 / 3: 6.144,
    1: 4.096,
    2: 2.048,
    4: 1.024,
    8: 0.512,
    16: 0.256,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show live ADS1115 readings and short-window noise statistics."
    )
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x48)
    parser.add_argument("--channel", type=int, choices=range(4), default=0)
    parser.add_argument("--gain", type=float, choices=FULL_SCALE_VOLTS, default=1)
    parser.add_argument(
        "--data-rate",
        type=int,
        choices=(8, 16, 32, 64, 128, 250, 475, 860),
        default=475,
    )
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--lines",
        action="store_true",
        help="print one new line per update instead of refreshing one terminal line",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    i2c = busio.I2C(board.SCL, board.SDA)
    adc = ADS.ADS1115(i2c, address=args.address)
    adc.gain = args.gain
    adc.data_rate = args.data_rate

    pins = (Pin.A0, Pin.A1, Pin.A2, Pin.A3)
    channel = AnalogIn(adc, pins[args.channel])
    samples = collections.deque(maxlen=args.window)
    full_scale = FULL_SCALE_VOLTS[args.gain]

    count = 0
    io_errors = 0
    rate_start = time.monotonic()
    measured_rate = 0.0
    next_display = rate_start

    print(
        f"ADS1115 address=0x{args.address:02x} channel=A{args.channel} "
        f"gain={args.gain:g} range=+/-{full_scale:g}V "
        f"configured_rate={args.data_rate}SPS"
    )
    print("Press Ctrl+C to stop.")

    try:
        while True:
            try:
                raw = channel.value
            except OSError as error:
                io_errors += 1
                print(
                    f"\nI2C read error #{io_errors}: {error}; retrying...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(0.25)
                continue

            samples.append(raw)
            count += 1
            now = time.monotonic()

            elapsed = now - rate_start
            if elapsed >= 1.0:
                measured_rate = count / elapsed
                count = 0
                rate_start = now

            if now < next_display:
                continue
            next_display = now + 0.1

            average = statistics.mean(samples)
            std_dev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
            peak_to_peak = max(samples) - min(samples)
            voltage = raw * full_scale / 32768.0
            low_nibble_count = len({value & 0x0F for value in samples})

            output = (
                f"raw={raw:6d}  voltage={voltage:7.4f}V  "
                f"mean={average:8.2f}  std={std_dev:6.2f}  "
                f"p-p={peak_to_peak:5d}  rate={measured_rate:6.1f}SPS  "
                f"low4={low_nibble_count:2d}/16  errors={io_errors}"
            )

            if args.lines:
                print(output, flush=True)
            else:
                print("\r" + output, end="", flush=True)
    except KeyboardInterrupt:
        if not args.lines:
            print()
        print("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
