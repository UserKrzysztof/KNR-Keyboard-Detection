from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from transformers import SamProcessor, SamModel, SamConfig
import matplotlib.pyplot as plt
import cv2
import time

os.chdir("/Users/krzysztof/machine_learning/object_detection/repo_january/KNR-Keyboard-Detection/integration")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def keyboard_bbox(image): #returns bounding box coordinates of keyboard
    model = YOLO("yolov8n.pt")  # replace with 'yolov8s.pt' for better accuracy

    result = model(image)
    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            if "keyboard" in label.lower():
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Convert BGR image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display the image using matplotlib
                plt.imshow(image_rgb)
                plt.title("Detected Keyboard")
                plt.axis('off')
                plt.show()
                return (x1, y1, x2, y2)

from patchify import patchify
from PIL import Image, ImageOps

patch_size = 256
step = 256

def make_square(image):
    image_pil = Image.fromarray(image)
    return np.array(ImageOps.pad(image_pil, (max(image_pil.size), max(image_pil.size)), color=0))

def transform_image(image, bbox):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = bbox
    image = image[y1:y2, x1:x2]
    image = make_square(image)
    image = cv2.resize(image, (512, 512))
    patches = patchify(image, (patch_size, patch_size), step=step)
    patches = patches.reshape(-1, patch_size, patch_size)
    print(patches.shape)
    return patches

def bbox_img(image, bbox):
    x1, y1, x2, y2 = bbox
    mask = np.zeros_like(image)
    mask[y1:y2, x1:x2] = 255
    mask = mask[y1:y2, x1:x2]
    mask = make_square(mask)
    mask = cv2.resize(mask, (512, 512))
    return mask

def mask_from_patches(patches, model, processor, input_points):
    patches = [Image.fromarray(patches).convert('RGB') for patches in patches]
    fig, axes = plt.subplots(1, len(patches), figsize=(15, 15))
    for ax, patch in zip(axes, patches):
        ax.imshow(patch)
        ax.axis('off')
    plt.show()
    mask_patches = []
    for patch in patches:
        inputs = processor(patch, return_tensors = 'pt', input_points = input_points)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, multimask_output = False)
        single_patch_mask = torch.sigmoid(outputs.pred_masks.squeeze(1))
        single_patch_mask = single_patch_mask.cpu().numpy().squeeze()
        single_patch_mask = (single_patch_mask > 0.5).astype(np.uint8)
        mask_patches.append(single_patch_mask)
    fig, axes = plt.subplots(1, len(mask_patches), figsize=(15, 15))
    for ax, patch in zip(axes, mask_patches):
        ax.imshow(patch)
        ax.axis('off')
    plt.show()
    mask = np.zeros((512, 512))
    for i in range(0, 512, step):
        for j in range(0, 512, step):
            mask[i:i+patch_size, j:j+patch_size] = mask_patches.pop(0)
    
    return mask

key_det_model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

key_det_model = SamModel(config=key_det_model_config)
key_det_model.load_state_dict(torch.load("key_detection_model.pth", map_location = torch.device("mps")))
key_det_model.to(device)  # Move the model to the same device as the input tensor

array_size = 256

# Define the size of your grid
grid_size = 10

# Generate the grid points
x = np.linspace(0, array_size-1, grid_size)
y = np.linspace(0, array_size-1, grid_size)

# Generate a grid of coordinates
xv, yv = np.meshgrid(x, y)

# Convert the numpy arrays to lists
xv_list = xv.tolist()
yv_list = yv.tolist()

# Combine the x and y coordinates into a list of list of lists
input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

#We need to reshape our nxn grid to the expected shape of the input_points tensor
# (batch_size, point_batch_size, num_points_per_image, 2),
# where the last dimension of 2 represents the x and y coordinates of each point.
#batch_size: The number of images you're processing at once.
#point_batch_size: The number of point sets you have for each image.
#num_points_per_image: The number of points in each set.
input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

import cv2
import time

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

bbox = None
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break  # Exit if camera fails

    if bbox is not None:
        print("Waiting 5 seconds before re-evaluating bbox...")
        time.sleep(5)  # Shorter wait time for efficiency

    new_bbox = keyboard_bbox(frame)
    if new_bbox is not None:
        print("New BBox detected:", new_bbox)
        print("Press 'y' to proceed with processing, or 'n' to skip to the next frame.")

        while True:
            key = cv2.waitKey(0) & 0xFF  # Wait for user input
            if key == ord('y'):
                bbox = new_bbox  # Accept new bbox
                break
            elif key == ord('n'):
                print("Skipping this frame...")
                bbox = None  # Reset bbox to skip processing
                break

    if bbox is not None:
        # Process the frame
        patches = transform_image(frame, bbox)
        mask = mask_from_patches(patches, key_det_model, processor, input_points)
        bbox_im = bbox_img(frame, bbox)

        print("Mask Shape:", mask.shape)
        print("BBox Image Shape:", bbox_im.shape)

        # Combine mask and bbox_im into a single picture with 3 classes
        combined_mask = np.zeros_like(bbox_im)
        combined_mask[np.all(bbox_im == [255, 255, 255], axis=-1)] = [0, 255, 0]  # Green for bbox
        combined_mask[mask > 0] = [255, 0, 0]  # Red for mask (handles non-binary masks)

        x1, y1, x2, y2 = bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1

        # Extract the green channel
        green_channel = combined_mask[:, :, 1]

        # Create a binary mask where green == 255
        binary_green_mask = (green_channel == 255).astype(np.uint8)

        # Find indices where green == 255
        y_indices, x_indices = np.where(binary_green_mask > 0)

        if len(y_indices) == 0 or len(x_indices) == 0:
            print("No valid green region found! Skipping overlay.")
        else:
            # Compute bounding box of the green area
            y_start, y_end = y_indices.min(), y_indices.max()
            x_start, x_end = x_indices.min(), x_indices.max()

            # Crop and resize mask
            cropped_mask = combined_mask[y_start:y_end+1, x_start:x_end+1, :]
            resized_mask = cv2.resize(cropped_mask, (bbox_width, bbox_height))

            # Prepare overlay
            overlay = np.zeros_like(frame)
            overlay[y1:y2, x1:x2] = resized_mask  

            # Blend overlay with frame
            alpha = 0.5
            frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)

            cv2.imshow('Camera', frame)
            print("Press Enter to continue...")
            if cv2.waitKey(0) == 13:
                bbox = None  # ASCII 13 is Enter key
                continue

    # Display the frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
