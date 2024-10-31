import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Import the U-2-Net model
from model import U2NET  # Adjust the import path if necessary

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# Load the pre-trained model
model = U2NET(3, 1)
model.load_state_dict(torch.load('saved_models/u2net/u2net.pth', map_location=torch.device('cpu')))
model.eval()

def remove_background(input_image_path, output_image_path):
    # Load the image
    image = Image.open(input_image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        d1, *_ = model(input_image)

    # Process the output mask
    pred = d1.squeeze().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize between 0 and 1
    mask = (pred > 0.5).astype(np.uint8) * 255  # Thresholding

    # Resize mask to original image size
    mask = Image.fromarray(mask).resize(image.size, resample=Image.BILINEAR)

    # Apply mask to the image
    image_np = np.array(image)
    mask_np = np.array(mask)[:, :, np.newaxis]
    result = np.concatenate([image_np, mask_np], axis=2)

    # Save the output image with alpha channel
    Image.fromarray(result).save(output_image_path)

# Example usage
if __name__ == '__main__':
    input_path = 'input.jpg'   # Replace with your input image path
    output_path = 'output.png' # Output image will have transparency
    remove_background(input_path, output_path)
