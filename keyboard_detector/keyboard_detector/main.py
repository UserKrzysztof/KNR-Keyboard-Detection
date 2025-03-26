import cv2
import time
import numpy as np
import os
import utils
import mask_finder
import key_finder
from matplotlib import pyplot

CAMERA_OR_FILE = 1#"input.mp4" #0

SENTENCE = "HAL-062"

os.chdir("/home/krzysztof/machine_learning/keyboard_detection/KNR-Keyboard-Detection/node")

#Setup models
mf = mask_finder.MaskFinder()#key_det_model_path=os.path.join("models","key_detection_model (1).pth"))
key_det_model = mf.key_det_model 
keyboard_bbox_model = mf.keyboard_bbox_model
processor = mf.processor 
input_points = mf.input_points
step = mf.step
patch_size = mf.patch_size
device = mf.device

kf = key_finder.KeyFinder(min_keys=50, 
                           probability_threshold=0.8,
                           min_key_size=1500,
                           key_displacement_distance=5e-2,
                           input_missing_keys=False,
                           use_gauss=False,
                           max_cluster_size=None,
                           check_space_eccentricity= True,
                           min_eccentricity = 0.2,
                           cluster_epsilon=np.inf)

# Open the default camera
cam = cv2.VideoCapture(2)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter()
out.open('output.avi', fourcc, 20.0, (frame_width, frame_height))

bbox = None
timer = None
check_time = False
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break  # Exit if camera fails

    if bbox is not None:
        print("Waiting 5 seconds before re-evaluating bbox...")
        time.sleep(5)  # Shorter wait time for efficiency

    
    if check_time is False:
        new_bbox = mask_finder.keyboard_bbox(frame, keyboard_bbox_model)
    else:
        now = time.time()
        if now - timer > 5:
            new_bbox = mask_finder.keyboard_bbox(frame, keyboard_bbox_model)
            check_time = False
        else:
            new_bbox = None

    if new_bbox is not None:
        cv2.imshow('Camera', frame)
        for i in range(40):
            out.write(frame)

        print("New BBox detected:", new_bbox)
        print("Press 'y' to proceed with processing, or 'n' to skip to the next frame.")

        while True:
            key = cv2.waitKey(0) & 0xFF  # Wait for user input
            if key == ord('y'):
                bbox = new_bbox  # Accept new bbox
                break
            elif key == ord('n'):
                print("Skipping this frame...")
                print("Waiting 10s before next detection")
                check_time = True
                timer = time.time()
                bbox = None  # Reset bbox to skip processing
                break

    if bbox is not None:
        # Process the frame
        print("Detecting keys")
        patches = mask_finder.transform_image(frame, bbox, patch_size, step)
        mask = mask_finder.mask_from_patches(patches, key_det_model, processor, input_points, patch_size, step, device)
        bbox_im = mask_finder.bbox_img(frame, bbox)

        print("Mask Shape:", mask.shape)
        print("BBox Image Shape:", bbox_im.shape)

        combined_mask = mask_finder.combine_mask(bbox_im, mask)

        frame, overlay = utils.get_frame_overlay(bbox, combined_mask, frame)

        keys = utils.retrieve_keys_from_mask(overlay)

        cv2.imshow('Camera', frame)
        for i in range(40):
            out.write(frame)

        print("Evaluating keys positions")
        try:
            saved_frame = frame.copy()
            letters = kf.find(keys)
            for key, value in letters.items():
                x, y = kf.original_coords(key, value, frame)

                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1) 

                text = str(key)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size
                
                bg_x1 = int(x) + 5
                bg_y1 = int(y) - 15
                bg_x2 = bg_x1 + text_width + 5
                bg_y2 = bg_y1 + text_height + 5
                
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), thickness=-1)
                cv2.putText(frame, text, (bg_x1, bg_y2 - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)  # Black text

            cv2.imshow('Camera', frame)
            time.sleep(3)
            for i in range(60):
                out.write(frame)

            frame = saved_frame
            cv2.putText(frame, f"Writing: {SENTENCE}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            cv2.imshow('Camera', frame)
            time.sleep(2)
            for i in range(40):
                out.write(frame)

            for key in SENTENCE:
                value = letters.get(key, None) 
                if value is None:
                    print(f"Key {key} wasnt found")
                    continue
                x, y = kf.original_coords(key, value, frame)
                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1) 
                cv2.imshow('Camera', frame)
                time.sleep(2)
                for i in range(40):
                    out.write(frame)

        except Exception as e:
            print("An error ocured")
            print(e)

        cv2.imshow('Camera', frame)
        for i in range(100):
            out.write(frame)
        print("Press Enter to continue...")
        if cv2.waitKey(0) == 13:
            bbox = None  # ASCII 13 is Enter key
            continue

    # Display the frame
    cv2.imshow('Camera', frame)
    out.write(frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
