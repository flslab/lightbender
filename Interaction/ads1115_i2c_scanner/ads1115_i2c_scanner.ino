#include <Wire.h>

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }

  Wire.begin();
  Wire.setWireTimeout(25000, true);
  Serial.println("Arduino Uno I2C scanner");
}

void loop() {
  byte found = 0;

  Serial.print("SDA(A4)=");
  Serial.print(digitalRead(SDA));
  Serial.print(" SCL(A5)=");
  Serial.println(digitalRead(SCL));
  Serial.println("Scanning I2C addresses 0x03-0x77...");
  for (byte address = 0x03; address <= 0x77; ++address) {
    Wire.clearWireTimeoutFlag();
    Wire.beginTransmission(address);
    const byte error = Wire.endTransmission();

    if (Wire.getWireTimeoutFlag()) {
      Serial.print("TIMEOUT at 0x");
      if (address < 0x10) {
        Serial.print('0');
      }
      Serial.println(address, HEX);
      break;
    }

    if (error == 0) {
      Serial.print("FOUND 0x");
      if (address < 0x10) {
        Serial.print('0');
      }
      Serial.println(address, HEX);
      ++found;
    } else if (error == 4) {
      Serial.print("Unknown error at 0x");
      if (address < 0x10) {
        Serial.print('0');
      }
      Serial.println(address, HEX);
    }
  }

  if (found == 0) {
    Serial.println("NO I2C DEVICES FOUND");
  } else {
    Serial.print("Device count: ");
    Serial.println(found);
  }

  Serial.println();
  delay(2000);
}
