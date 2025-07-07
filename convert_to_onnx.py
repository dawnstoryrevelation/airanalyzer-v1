# ===================================================================================
# || AIR ANALYZER V1 - PYTORCH TO ONNX CONVERSION SCRIPT
# ||
# || Purpose: Converts the trained PyTorch model (.pth) into the ONNX format (.onnx)
# ||          for use in web applications or other deployment environments.
# ||
# || Author: Master Coder for Air Analyzer Project
# || Version: 1.1.0 (Now with post-creation verification)
# ===================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os # NEW: Imported the 'os' module for file system operations

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

# ===================================================================================
# || STEP 1: DEFINE THE EXACT MODEL ARCHITECTURE
# || This class must be an *exact, line-for-line copy* of your training script.
# ===================================================================================
class AirPollutionCNN(nn.Module):
    """The CNN architecture for the Air Analyzer model."""
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
        """The forward pass of the model."""
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = F.relu(self.bn4(self.conv4(x))); x = self.pool3(x)
        x = F.relu(self.bn5(self.conv5(x))); x = self.pool4(x)
        x = self.global_avg_pool(x); x = x.view(x.size(0), -1)
        x = self.dropout1(x); x = F.relu(self.fc1(x)); x = self.dropout2(x)
        x = F.relu(self.fc2(x)); x = self.dropout3(x); x = self.fc3(x)
        return x

# ===================================================================================
# || STEP 2: CONFIGURE AND EXECUTE THE CONVERSION
# ===================================================================================
def convert_model_to_onnx():
    """Main function to load and convert the model, now with verification."""
    pytorch_model_path = 'air_analyzer_cnn_iden_7m.pth'
    onnx_model_path = 'air_analyzer_cnn_iden_7m.onnx'
    num_classes = 6
    image_size = 224

    # NEW: Get the absolute path for clear logging
    absolute_onnx_path = os.path.abspath(onnx_model_path)

    print(f"[*] 1. Instantiating the '{AirPollutionCNN.__name__}' model architecture...")
    model = AirPollutionCNN(num_classes=num_classes)
    print("    - Model architecture created successfully.")

    print(f"[*] 2. Loading trained weights from '{pytorch_model_path}'...")
    try:
        checkpoint = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("    - Successfully loaded weights.")
    except Exception as e:
        print(f"[!] ERROR: Failed to load weights from '{pytorch_model_path}'. Aborting. Error: {e}")
        return

    model.eval()
    print("[*] 3. Setting model to evaluation mode (model.eval()).")
    
    dummy_input = torch.randn(1, 3, image_size, image_size)
    print(f"[*] 4. Creating a dummy input tensor of shape: {dummy_input.shape}")

    print(f"[*] 5. Exporting the model to ONNX format at '{absolute_onnx_path}'...")
    try:
        torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        print("    - ONNX export command executed.")
    except Exception as e:
        print(f"[!] ERROR: The ONNX export process failed with an exception: {e}")
        return

    # ===================================================================================
    # || NEW: STEP 6 - VERIFY FILE CREATION
    # ===================================================================================
    print("[*] 6. Verifying that the output file was created on disk...")
    if os.path.exists(absolute_onnx_path):
        print("\n" + "="*60)
        print("✅ SUCCESS: Model has been converted AND verified!")
        print(f"   Your web-ready model is saved at: {absolute_onnx_path}")
        print("   You can now use this file with your 'airanalyzercnn.html' application.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ CRITICAL FAILURE: The script reported success, but the file was NOT found on disk.")
        print(f"   Expected location: {absolute_onnx_path}")
        print("\n   This is likely due to an issue with your environment:")
        print("   1. Permissions: The script may not have rights to write to this folder.")
        print("   2. Cloud Sync (OneDrive/Dropbox): These services can interfere with file creation.")
        print("\n   RECOMMENDED ACTION: Move the project to a simple, local folder (e.g., 'C:\\dev\\project') and run this script again.")
        print("="*60)

if __name__ == '__main__':
    convert_model_to_onnx()