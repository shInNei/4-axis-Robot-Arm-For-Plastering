import sys
import serial
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Replace with your Arduino's COM port
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

class ServoController(QWidget):
    def __init__(self):
        super().__init__()
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        self.language = "English"
        self.angle_limits = [180, 70, 60, 180]  # Max angle limits for joints
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Arduino Servo Controller")
        self.setGeometry(100, 100, 500, 300)

        # Language selection dropdown
        self.languageSelector = QComboBox()
        self.languageSelector.addItems(["English", "日本語"])
        self.languageSelector.currentTextChanged.connect(self.changeLanguage)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.languageSelector)

        # Servo control layout
        self.control_layout = QVBoxLayout()
        self.sliders = []
        self.inputBoxes = []
        self.labels = []
        
        for i in range(4):
            h_layout = QHBoxLayout()
            
            # Joint Label
            label = QLabel(f"Joint {i + 1}:")
            self.labels.append(label)
            
            # Min and Max Labels
            left_limit = QLabel("0")
            right_limit = QLabel(str(self.angle_limits[i]))
            
            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, self.angle_limits[i])
            slider.setValue(self.angle_limits[i] // 2)
            slider.valueChanged.connect(self.update_angle)
            self.sliders.append(slider)
            
            # Input Box
            inputBox = QLineEdit()
            inputBox.setText(str(slider.value()))
            inputBox.setMaxLength(3)
            inputBox.textChanged.connect(self.update_slider)
            self.inputBoxes.append(inputBox)
            
            h_layout.addWidget(label)
            h_layout.addWidget(left_limit)
            h_layout.addWidget(slider)
            h_layout.addWidget(right_limit)
            h_layout.addWidget(inputBox)
            
            self.control_layout.addLayout(h_layout)
        
        # Buttons
        self.sendButton = QPushButton("Send Angles")
        self.sendButton.clicked.connect(self.send_angles)
        self.control_layout.addWidget(self.sendButton)
        
        self.resetButton = QPushButton("Reset Servos")
        self.resetButton.clicked.connect(self.reset_servos)
        self.control_layout.addWidget(self.resetButton)

        main_layout.addLayout(self.control_layout)
        self.setLayout(main_layout)

    def changeLanguage(self, text):
        if text == "日本語":
            self.setFont(QFont("SmartFontUI", 12))
            for i, label in enumerate(self.labels):
                label.setText(f"関節 {i + 1}:")
            self.sendButton.setText("送信")
            self.resetButton.setText("リセット")
        else:
            self.setFont(QFont("Roboto", 12))
            for i, label in enumerate(self.labels):
                label.setText(f"Joint {i + 1}:")
            self.sendButton.setText("Send Angles")
            self.resetButton.setText("Reset Servos")

    def update_angle(self):
        sender_slider = self.sender()
        index = self.sliders.index(sender_slider)
        self.inputBoxes[index].setText(str(sender_slider.value()))

    def update_slider(self):
        sender_input = self.sender()
        index = self.inputBoxes.index(sender_input)
        try:
            new_value = int(sender_input.text())
            if 0 <= new_value <= self.angle_limits[index]:
                self.sliders[index].setValue(new_value)
            else:
                sender_input.setText(str(self.sliders[index].value()))  # Reset invalid input
        except ValueError:
            sender_input.setText(str(self.sliders[index].value()))  # Reset invalid input

    def send_angles(self):
        angles = [str(slider.value()) for slider in self.sliders]
        formatted_angles = " ".join(angles)
        self.ser.write((formatted_angles + "\n").encode())
        self.ser.flush()

    def reset_servos(self):
        self.ser.write(b"reset\n")
        self.ser.flush()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ServoController()
    window.show()
    sys.exit(app.exec())



