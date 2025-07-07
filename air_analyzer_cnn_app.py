#!/usr/bin/env python3
"""
🌍 Air Quality CNN Analyzer - GUI Application 🌍

This application provides a graphical user interface to interact with the
trained Air Pollution CNN model. Users can either upload an image or
take a picture with their webcam to get a real-time air quality prediction.
"""

import tkinter
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2 # OpenCV for camera access
import os

# --- MODEL AND CONFIGURATION (Copied from your training script) ---
# This section ensures the app can load and use your model without
# needing the original training script.

# 🎯 CONFIGURATION
MODEL_CONFIG = {
    'NUM_CLASSES': 6,
    'IMG_SIZE': 224,
    'POLLUTION_LEVELS': {
        0: "Good 😊",
        1: "Moderate 🙂",
        2: "Unhealthy for Sensitive Groups 😐",
        3: "Unhealthy 😕",
        4: "Very Unhealthy 😨",
        5: "Hazardous 😷"
    },
    'AQI_RANGES': {
        0: "0-50 AQI",
        1: "51-100 AQI",
        2: "101-150 AQI",
        3: "151-200 AQI",
        4: "201-300 AQI",
        5: "300+ AQI"
    },
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# 🏗️ CNN ARCHITECTURE (~10M PARAMETERS)
class AirPollutionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(AirPollutionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

# 🔄 IMAGE TRANSFORMS (Must match validation transform from training)
val_transform = transforms.Compose([
    transforms.Resize((MODEL_CONFIG['IMG_SIZE'], MODEL_CONFIG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GUI APPLICATION CLASS ---

class AirAnalyzerApp(ctk.CTk):
    def __init__(self, model, model_path):
        super().__init__()

        self.title("Air Quality CNN Analyzer")
        self.geometry("900x650")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.model = model
        self.transform = val_transform
        self.device = MODEL_CONFIG['DEVICE']
        self.model_path = model_path
        
        self.current_pil_image = None
        self.camera_active = False
        self.cap = None

        # --- Layout Configuration ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Widgets ---
        self.create_widgets()

    def create_widgets(self):
        # Header
        self.header_label = ctk.CTkLabel(self, text="🌍 Air Quality CNN Analyzer 🌍", font=ctk.CTkFont(size=24, weight="bold"))
        self.header_label.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

        # Image Display Frame
        self.image_frame = ctk.CTkFrame(self, corner_radius=15)
        self.image_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="Upload or capture an image to begin", font=ctk.CTkFont(size=16))
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Controls and Results Frame
        self.controls_frame = ctk.CTkFrame(self, corner_radius=15)
        self.controls_frame.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        
        # Action Buttons
        self.upload_button = ctk.CTkButton(self.controls_frame, text="Upload Image", command=self.upload_image, font=ctk.CTkFont(size=14))
        self.upload_button.pack(pady=15, padx=20, fill="x")

        self.camera_button = ctk.CTkButton(self.controls_frame, text="Use Camera", command=self.toggle_camera, font=ctk.CTkFont(size=14))
        self.camera_button.pack(pady=15, padx=20, fill="x")
        
        # Results Display
        self.results_label = ctk.CTkLabel(self.controls_frame, text="Prediction Results", font=ctk.CTkFont(size=18, weight="bold", underline=True))
        self.results_label.pack(pady=(20, 10))
        
        # Prediction
        self.result_level_label = ctk.CTkLabel(self.controls_frame, text="Level: ---", font=ctk.CTkFont(size=16))
        self.result_level_label.pack(pady=5, padx=20, anchor="w")
        
        # AQI Range
        self.result_aqi_label = ctk.CTkLabel(self.controls_frame, text="AQI Range: ---", font=ctk.CTkFont(size=16))
        self.result_aqi_label.pack(pady=5, padx=20, anchor="w")
        
        # Confidence
        self.result_confidence_label = ctk.CTkLabel(self.controls_frame, text="Confidence: ---", font=ctk.CTkFont(size=16))
        self.result_confidence_label.pack(pady=5, padx=20, anchor="w")

        # Status Bar
        self.status_label = ctk.CTkLabel(self, text=f"Model loaded: {os.path.basename(self.model_path)} on {str(self.device).upper()}", text_color="gray")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="w")

    def upload_image(self):
        if self.camera_active:
            self.stop_camera()
            
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if not filepath:
            return

        self.current_pil_image = Image.open(filepath).convert("RGB")
        self.display_image(self.current_pil_image)
        self.predict()

    def toggle_camera(self):
        if not self.camera_active:
            self.start_camera()
        else:
            self.capture_image()
            
    def start_camera(self):
        self.cap = cv2.VideoCapture(0) # Use camera 0
        if not self.cap.isOpened():
            self.image_label.configure(text="Error: Could not open camera.")
            self.cap = None
            return
        
        self.camera_active = True
        self.camera_button.configure(text="Take Picture", fg_color="red")
        self.upload_button.configure(state="disabled")
        self.update_camera_feed()

    def update_camera_feed(self):
        if not self.camera_active:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Convert OpenCV BGR frame to PIL RGB image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            self.display_image(pil_image)
            self.after(15, self.update_camera_feed) # Update ~60fps
        else:
            self.stop_camera()

    def capture_image(self):
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.stop_camera()
            self.display_image(self.current_pil_image)
            self.predict()
            
    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.camera_active = False
        self.camera_button.configure(text="Use Camera", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.upload_button.configure(state="normal")
        
    def display_image(self, pil_image):
        # Resize image to fit the label while maintaining aspect ratio
        img_w, img_h = pil_image.size
        label_w, label_h = 450, 450 # Approx size of the image label
        
        ratio = min(label_w / img_w, label_h / img_h)
        new_size = (int(img_w * ratio), int(img_h * ratio))
        
        resized_image = pil_image.resize(new_size, Image.LANCZOS)
        
        ctk_image = ctk.CTkImage(light_image=resized_image, dark_image=resized_image, size=new_size)
        self.image_label.configure(image=ctk_image, text="")

    def predict(self):
        if not self.current_pil_image:
            return

        # Preprocess the image
        img_tensor = self.transform(self.current_pil_image).unsqueeze(0).to(self.device)
        
        # Make a prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = predicted_idx.item()
            confidence_score = confidence.item()

        # Get results from config
        level = MODEL_CONFIG['POLLUTION_LEVELS'][predicted_class]
        aqi = MODEL_CONFIG['AQI_RANGES'][predicted_class]

        # Update the GUI with results
        self.result_level_label.configure(text=f"Level: {level}")
        self.result_aqi_label.configure(text=f"AQI Range: {aqi}")
        self.result_confidence_label.configure(text=f"Confidence: {confidence_score:.2%}")


def main():
    # --- MODEL LOADING ---
    MODEL_FILE = "air_analyzer_cnn_iden_7m.pth"
    
    if not os.path.exists(MODEL_FILE):
        print(f"FATAL ERROR: Model file '{MODEL_FILE}' not found.")
        print("Please ensure the trained model file is in the same directory as this script.")
        return

    # Instantiate the model
    model = AirPollutionCNN(num_classes=MODEL_CONFIG['NUM_CLASSES'])

    # Load the state dictionary
    try:
        # Load the entire checkpoint
        checkpoint = torch.load(MODEL_FILE, map_location=MODEL_CONFIG['DEVICE'])
        
        # Your training script saves the model in a dictionary with a key 'model_state_dict'.
        # We need to load that specific state dict.
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback for if the file only contains the state_dict
            model.load_state_dict(checkpoint)

    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        print("The model architecture in this script might not match the one in the saved file.")
        return
        
    model.to(MODEL_CONFIG['DEVICE'])
    model.eval() # Set model to evaluation mode

    # --- RUN APP ---
    app = AirAnalyzerApp(model=model, model_path=MODEL_FILE)
    app.mainloop()


if __name__ == "__main__":
    # Advise user on dependencies if they are missing
    try:
        import customtkinter
        import PIL
        import cv2
        import torch
    except ImportError as e:
        print("="*60)
        print("ERROR: A required library is missing.")
        print(f"Missing library: {e.name}")
        print("\nPlease install the required packages by running:")
        print("pip install customtkinter pillow opencv-python torch torchvision")
        print("="*60)
    else:
        main()