import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def get_frame_overlay(bbox, combined_mask, frame, alpha = 0.5):
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
        return frame, None
    
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
    frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)

    return frame, overlay

def retrieve_keys_from_mask(mask):
    mask = np.transpose(mask, (2, 0, 1))
    binary_mask = (mask[1, :] > 0).astype(np.uint8)

    y_indices, x_indices = np.where(binary_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("No positive values found in mask[2, :]")

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    full_size_keys = np.zeros_like(mask[0, :], dtype=np.uint8)
    cropped_region = (mask[0, y_min:y_max+1, x_min:x_max+1] > 0).astype(np.uint8)

    full_size_keys[y_min:y_max+1, x_min:x_max+1] = cropped_region

    return full_size_keys
