import os
from ultralytics import YOLO
from transformers import SamProcessor, SamModel, SamConfig
import torch
import cv2
import numpy as np
from patchify import patchify
from PIL import Image, ImageOps

class MaskFinder():
    def __init__(self, 
                 keyboard_bbox_model_path = None, 
                 key_det_model_path = None,
                 patch_size = 256,
                 step = 256,
                 array_size = 256,
                 grid_size = 10):
        if key_det_model_path is None:
            key_det_model_path = os.path.join(".","src","keyboard_detector","keyboard_detector","models","key_detection_model.pth")
        if keyboard_bbox_model_path is None:
            keyboard_bbox_model = os.path.join(".","src","keyboard_detector","keyboard_detector","models", "yolov8n.pt")

        assert os.path.exists(key_det_model_path), f"The path {key_det_model_path} does not exist"
        assert os.path.exists(keyboard_bbox_model), f"The path {keyboard_bbox_model} does not exist"
        assert type(patch_size) is int
        assert type(step) is int
        assert type(array_size) is int
        assert type(grid_size) is int

        self._setup(keyboard_bbox_model, 
                    key_det_model_path,
                    patch_size,
                    step,
                    array_size,
                    grid_size)  

    def _setup(self, 
               keyboard_bbox_model_path, 
               key_det_model_path,
               patch_size,
               step,
               array_size,
               grid_size):
        print("Setting up model for keyboard detection", end="...")
        self.keyboard_bbox_model = YOLO(keyboard_bbox_model_path)
        print("Success")

        print("Setting up device", end="...")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #else  torch.device("mps") if torch.backends.mps.is_available()
        print("Device:", self.device)

        print("Setting up model for keys detection", end="...")
        key_det_model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        self.key_det_model = SamModel(config=key_det_model_config)
        self.key_det_model.load_state_dict(torch.load(key_det_model_path, map_location = self.device))
        self.key_det_model.to(self.device) 
        print("Success")

        print("Setting up additional params", end="...")
        self.patch_size = patch_size
        self.step = step

        # Generate the grid points
        x = np.linspace(0, array_size-1, grid_size)
        y = np.linspace(0, array_size-1, grid_size)

        # Generate a grid of coordinates
        xv, yv = np.meshgrid(x, y)

        # Convert the numpy arrays to lists
        xv_list = xv.tolist()
        yv_list = yv.tolist()

        # Combine the x and y coordinates into a list of list of lists
        self.input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

        #We need to reshape our nxn grid to the expected shape of the input_points tensor
        # (batch_size, point_batch_size, num_points_per_image, 2),
        # where the last dimension of 2 represents the x and y coordinates of each point.
        #batch_size: The number of images you're processing at once.
        #point_batch_size: The number of point sets you have for each image.
        #num_points_per_image: The number of points in each set.
        self.input_points = torch.tensor(self.input_points).view(1, 1, grid_size*grid_size, 2)
        print("Success")


def keyboard_bbox(image, model): #returns bounding box coordinates of keyboard
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
                return (x1, y1, x2, y2)

def make_square(image):
    image_pil = Image.fromarray(image)
    return np.array(ImageOps.pad(image_pil, (max(image_pil.size), max(image_pil.size)), color=0))

def transform_image(image, bbox, patch_size, step):
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

def mask_from_patches(patches, model, processor, input_points, patch_size, step, device):
    patches = [Image.fromarray(patches).convert('RGB') for patches in patches]

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
    
    mask = np.zeros((512, 512))
    for i in range(0, 512, step):
        for j in range(0, 512, step):
            mask[i:i+patch_size, j:j+patch_size] = mask_patches.pop(0)
    return mask

def combine_mask(bbox_im, mask):
    # Combine mask and bbox_im into a single picture with 3 classes
    combined_mask = np.zeros_like(bbox_im)
    combined_mask[np.all(bbox_im == [255, 255, 255], axis=-1)] = [0, 255, 0]  # Green for bbox
    combined_mask[mask > 0] = [255, 0, 0]  # Red for mask (handles non-binary masks)
    return combined_mask

