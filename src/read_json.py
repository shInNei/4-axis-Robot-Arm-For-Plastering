import json
import serial
import time

# Configuration
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
JSON_FILE = "sequence_542_steps.json"  # Change to your actual JSON file
TIMEOUT = 5  # Maximum time to wait for Arduino's response (seconds)

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(3)

def send_action(action):
    formatted_action = " ".join(map(str, action)) + "\n"
    ser.write(formatted_action.encode())  # Send action
    ser.flush()
    print(f"Sent: {formatted_action.strip()}")

    # Wait for acknowledgment
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        response = ser.readline().decode().strip()
        if response == "OK":  # Adjust if your Arduino sends a different response
            print("Arduino acknowledged, moving to next action.")
            return
        time.sleep(0.1)  # Small wait before checking again

    print("Warning: No response from Arduino, moving to next action.")

def main():
    # Load JSON file
    with open(JSON_FILE, "r") as file:
        data = json.load(file)
    
    # Iterate through actions and send them
    for action in data["actions"]:
        send_action(action)
    
    print("All actions sent.")

if __name__ == "__main__":
    main()
    ser.close()

