#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  110 // Minimum pulse length count (out of 4096)
#define SERVOMAX  500 // Maximum pulse length count (out of 4096)
#define SERVO_FREQ 50  // Analog servos run at ~50 Hz updates

uint16_t currentPulse[4] = {SERVOMIN, SERVOMIN, SERVOMIN, SERVOMIN};

void setup() {
  delay(1000);
  Serial.begin(9600);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);
  // resetServos();
  // Serial.println("Enter angles in format: x y z w or type 'reset' to reset servos");
}

void moveServo(uint8_t motorID, uint16_t desiredAngle) {
  if (motorID > 3) return; // Prevent out-of-bounds access
  
  uint16_t servoMax = (motorID == 3) ? 415 : SERVOMAX;
  uint16_t desiredPulse = map(desiredAngle, 0, 180, SERVOMIN, servoMax);
  
  if (desiredPulse < SERVOMIN || desiredPulse > servoMax) {
    Serial.print("Invalid angle for motor ");
    Serial.print(motorID);
    Serial.println("! Enter a value between 0 and 180.");
    return;
  }

  while (currentPulse[motorID] != desiredPulse) {
    if (currentPulse[motorID] > desiredPulse) {
      currentPulse[motorID]--;
    } else {
      currentPulse[motorID]++;
    }
    pwm.setPWM(motorID, 0, currentPulse[motorID]);
    delay(7.5); // Small delay for smooth movement
  }
}

void resetServos() {
  Serial.println("Resetting all servos to SERVOMIN...");
  for (int i = 0; i < 4; i++) {
    moveServo(i, 0); // Moves all servos to 0 degrees (SERVOMIN)
    // delay(200);
  }
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.equalsIgnoreCase("reset")) {
      resetServos();
      Serial.println("Finish Reset!");
      return;
    }
    
    int angles[4];
    int index = 0;
    char *ptr = strtok((char*)input.c_str(), " ");
    while (ptr != NULL && index < 4) {
      angles[index++] = atoi(ptr);
      ptr = strtok(NULL, " ");
    }

    if (index == 4) { // Ensure we got exactly 4 values
      for (int i = 0; i < 4; i++) {
        moveServo(i, angles[i]);
        // delay(100);
      }
      Serial.println("OK");
      // delay(200);
    } else {
      Serial.println("Invalid input! Enter 4 angles separated by spaces or type 'reset'.");
    }
  }
}

