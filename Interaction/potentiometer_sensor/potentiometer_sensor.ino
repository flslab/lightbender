// Potentiometer distance measurement with low-pass filtering and
// multi-point piecewise-linear calibration.
//
// Typical wiring:
//   Red    -> Arduino 5V (or 3.3V on a 3.3V board)
//   Green  -> Arduino GND
//   Yellow -> Arduino A0
//
// Wire colors are not guaranteed. The signal wire must be the wiper of the
// potentiometer. Swapping Red and Green only reverses the measurement direction.

const uint8_t POT_PIN = A0;

// ADC settings.
const int ADC_MAX_VALUE = 1023;

// Measured calibration points. Raw ADC values decrease as distance increases.
// ADC 1023 corresponds to 0 mm (fully released/reference position).
const uint8_t CALIBRATION_COUNT = 11;

const float RAW_CALIBRATION[CALIBRATION_COUNT] = {
  1023.0f, 987.0f, 974.0f, 965.0f, 953.0f, 946.0f,
  935.0f, 925.0f, 876.0f, 665.0f, 500.0f
};

const float DISTANCE_MM[CALIBRATION_COUNT] = {
  0.0f, 1.3f, 2.2f, 3.3f, 4.8f, 5.6f,
  6.6f, 7.1f, 8.7f, 9.7f, 10.4f
};

// Exponential low-pass filter. Larger values respond faster but filter less.
// A practical range is approximately 0.05 to 0.30.
const float FILTER_ALPHA = 0.50f;

// Sampling/output interval. 20 ms = 50 samples per second.
const unsigned long SAMPLE_INTERVAL_MS = 20;

float filteredPotValue = 0.0f;
unsigned long lastSampleTime = 0;

// Estimate the Nano supply using the AVR's internal nominal 1.1 V reference.
// Absolute accuracy depends on the chip's bandgap tolerance, but fast changes
// reliably reveal supply droop or a brownout precursor during motor startup.
unsigned long readSupplyMillivolts() {
#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  delayMicroseconds(250);
  ADCSRA |= _BV(ADSC);
  while (bit_is_set(ADCSRA, ADSC)) {
  }
  const uint8_t low = ADCL;
  const uint8_t high = ADCH;
  const unsigned int result = (high << 8) | low;
  if (result == 0) {
    return 0;
  }
  return 1125300UL / result;
#else
  // This sketch is calibrated for a classic 5 V Nano. Unsupported boards
  // retain a useful nominal field rather than failing compilation.
  return 5000UL;
#endif
}

float computePosition(float rawValue) {
  // Clamp readings outside the calibrated range. ADC 1023 is the 0 mm
  // reference and is also the upper limit of the Uno's 10-bit ADC.
  if (rawValue >= RAW_CALIBRATION[0]) {
    return DISTANCE_MM[0];
  }
  if (rawValue <= RAW_CALIBRATION[CALIBRATION_COUNT - 1]) {
    return DISTANCE_MM[CALIBRATION_COUNT - 1];
  }

  // Find the two measured points surrounding the current reading, then
  // interpolate only within that interval. This follows the measured
  // nonlinearity without the endpoint oscillation of a high-order polynomial.
  for (uint8_t i = 0; i < CALIBRATION_COUNT - 1; ++i) {
    const float rawHigh = RAW_CALIBRATION[i];
    const float rawLow = RAW_CALIBRATION[i + 1];

    if (rawValue <= rawHigh && rawValue >= rawLow) {
      const float fraction = (rawHigh - rawValue) / (rawHigh - rawLow);
      return DISTANCE_MM[i]
           + fraction * (DISTANCE_MM[i + 1] - DISTANCE_MM[i]);
    }
  }

  // This should be unreachable because values outside the table are clamped.
  return DISTANCE_MM[CALIBRATION_COUNT - 1];
}

void setup() {
  Serial.begin(115200);

  // Initialize the filter with a real measurement. This avoids a false ramp
  // from zero during the first few readings.
  const int firstReading = analogRead(POT_PIN);
  filteredPotValue = firstReading;
  lastSampleTime = millis();

  Serial.println(
    "time_ms,raw,filtered,voltage,distance_mm,supply_voltage"
  );
}

void loop() {
  const unsigned long now = millis();

  // Unsigned subtraction remains valid when millis() wraps around.
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleTime = now;

  const float supplyVoltage = readSupplyMillivolts() / 1000.0f;

  // The supply measurement selects the internal bandgap channel. Discard the
  // first A0 conversion after switching back so it cannot bias the sensor.
  analogRead(POT_PIN);
  const int rawPotValue = analogRead(POT_PIN);

  filteredPotValue = FILTER_ALPHA * rawPotValue
                   + (1.0f - FILTER_ALPHA) * filteredPotValue;

  const float voltage = filteredPotValue
                      * supplyVoltage / ADC_MAX_VALUE;
  const float distanceMm = computePosition(filteredPotValue);

  // CSV output for Serial Monitor or Serial Plotter.
  Serial.print(now);
  Serial.print(',');
  Serial.print(rawPotValue);
  Serial.print(',');
  Serial.print(filteredPotValue, 2);
  Serial.print(',');
  Serial.print(voltage, 3);
  Serial.print(',');
  Serial.print(distanceMm, 3);
  Serial.print(',');
  Serial.println(supplyVoltage, 3);
}
