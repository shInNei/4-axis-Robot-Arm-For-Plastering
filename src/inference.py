import torch
import cv2
import numpy as np
import depthai as dai
import os
import time
from new_model import RobotArmMovement

# Set device
m_device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = RobotArmMovement().to(m_device)
model.load_state_dict(torch.load("modelV1_n.pth", map_location=m_device))
model.eval()

# Min and max values for first 3 elements
min_values = [-10, -10, -120]
max_values = [90, 90, 0.5]

# Ensure the folder exists
os.makedirs("test_f", exist_ok=True)

# Initial movement array
movement_array = [
    0.13212681, 80.20945, -10, 
    1, 0, 0, 0, 1, 0, 0, 0, 1
]

# Create pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(160, 120)  # Match model input size
cam.setInterleaved(False)
cam.setFps(30)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

# Define interval for stable execution
interval = 0.00833  # 8.33ms per iteration
last_time = time.time()

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device, open("predictions.txt", "w") as file:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    step = 0
    
    while True:  # Keep running until 'q' is pressed
        while time.time() - last_time < interval:
            pass  # Wait for the next cycle
        frame = q.get().getCvFrame()
        
        # Resize frame to 160x120
        frame = cv2.resize(frame, (160, 120))
        
        # Save frame
        image_path = f"test_f/frame_{step:05d}.png"
        cv2.imwrite(image_path, frame)
        
        # Preprocess image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize to [0,1]
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        
        # Convert image to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(m_device)
        
        # Normalize first three values
        normalized_movement = movement_array[:]
        for i in range(3):
            normalized_movement[i] = (normalized_movement[i] - min_values[i]) / (max_values[i] - min_values[i])
        
        movement_tensor = torch.tensor(normalized_movement, dtype=torch.float32).unsqueeze(0).to(m_device)
        
        # Run inference
        with torch.no_grad():
            predicted_tensor = model(image_tensor, movement_tensor)
        
        predicted_array = predicted_tensor.cpu().numpy().squeeze()
        
        # Unnormalize first three values
        for i in range(3):
            predicted_array[i] = predicted_array[i] * (max_values[i] - min_values[i]) + min_values[i]
        
        # Process remaining values
        for i in range(3, len(predicted_array)):
            predicted_array[i] = 1 if predicted_array[i] > 0.5 else -1 if predicted_array[i] < -0.5 else 0
        
        movement_array = predicted_array.tolist()
        
        # Write predictions to file
        file.write(f"Step {step + 1}: {predicted_array.tolist()}\n")
        
        # Display frame
        cv2.imshow("OAK Camera Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
        step += 1
    
    cv2.destroyAllWindows()

print("Predictions saved to predictions.txt")
