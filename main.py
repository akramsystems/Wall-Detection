#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy.ndimage import maximum_filter
import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.getcwd())
from floortrans.models import get_model
from floortrans.post_prosessing import split_prediction

def load_model():
    """Load the pre-trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model('hg_furukawa_original', 51)
    
    n_classes = 44
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    
    checkpoint = torch.load('./pkl_file/model_best_val_loss_var.pkl', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, device

def extract_wall_corners(heatmaps, threshold=0.1, nms_size=3):
    """Extract wall corner coordinates from heatmaps."""
    wall_corners = []
    for c in range(13):  # channels 0-12 for wall junctions
        channel = heatmaps[c]
        max_filt = maximum_filter(channel, size=nms_size)
        peaks = (channel == max_filt) & (channel > threshold)
        ys, xs = np.where(peaks)
        for x, y in zip(xs, ys):
            wall_corners.append((x, y, c))
    return wall_corners

def plot_joints_on_image(image, wall_corners):
    """Display image with joints marked."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for x, y, c in wall_corners:
        plt.plot(x, y, 'o', color='red', markersize=6)
    plt.title("Wall Joints/Corners")
    plt.axis('off')
    plt.show()

def main(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    print(f"Processing image: {image_path}")
    
    # Load model
    model, device = load_model()
    
    # Load and preprocess image
    img = Image.open(image_path)
    print(f"Image size: {img.size}")
    preprocess = transforms.Compose([transforms.ToTensor()])
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        pred = model(image_tensor)
    
    # Extract heatmaps
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    img_size = (height, width)
    split = [21, 12, 11]
    heatmaps, _, _ = split_prediction(pred, img_size, split)
    
    # Extract wall corners/joints
    joints = extract_wall_corners(heatmaps)
    
    # Display results
    plot_joints_on_image(img, joints)
    print(f"Found {len(joints)} wall joints/corners")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect wall joints/corners in floor plan images')
    parser.add_argument('image', nargs='?', default='test_image.png', 
                       help='Path to the image file (default: test_image.png)')
    
    args = parser.parse_args()
    main(args.image) 