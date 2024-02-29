from ultralytics import YOLO
import sys

from llava_descriptor import extract_images_from_grid


def yolo_filtering(image, class_name, model_or_path='yolov8l-world.pt'):
    
    model = YOLO(model_or_path) 
    model.set_classes(class_name)

    extracted_imgs = extract_images_from_grid(image, 2, 5)
    no_detection = []

    for i in range(10):
        res = model.predict(extracted_imgs[i], conf=0.1)

        if not res[0].boxes.xyxy.cpu().numpy().any():
            no_detection.append(i + 1)
    
    if not no_detection:
        print(f"The given class {class_name} has been detected in all 10 images")
    
    return no_detection
