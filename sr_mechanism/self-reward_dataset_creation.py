# python sr_mechanism/self-reward_dataset_creation.py --image_folder 'generative_images2'  --output_folder './optimal_pairs/'

import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from llava_descriptor import llava_loader, extract_images_from_grid
import os
from ultralytics import YOLO

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="Self-reward dataset creation script.")
parser.add_argument("--image_folder", required=True, help="Path to the image folder.")
parser.add_argument("--prompts_file", required=True, help="Path to the prompts folder.")
parser.add_argument("--llava_model", default="llava-hf/llava-1.5-7b-hf", help="Llava model ID.")
parser.add_argument("--yolo_model", default="yolov8x-worldv2.pt", help="Yolo model ID.")
parser.add_argument("--output_folder", required=True, help="Path to the output folder.")
args = parser.parse_args()

# Set variables from command line arguments
image_folder_path = args.image_folder
prompts_file_path = args.prompts_file
llava_model_id = args.llava_model
yolo_model_id = args.yolo_model
save_path = args.output_folder


model_yolo = YOLO(yolo_model_id) 
model_llava, processor_llava = llava_loader(llava_model_id)

image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for selected_image in image_files:
    image = Image.open(os.path.join(image_folder_path, selected_image))
    print(f"processing image {selected_image}:")

    extracted_imgs = extract_images_from_grid(image, 2, 5)

    image_number = selected_image.split('_')[0]
    class_name = selected_image.split('_')[1].split('.')[0]

    with open(prompts_file_path, 'r', encoding='utf-8') as prompts_file:
        lines = prompts_file.readlines()
        if int(image_number) < len(lines):
            prompt_line = lines[int(image_number)-1].strip()
            description = prompt_line.split(';')[1].strip()
            
    SYST_PROMPT = f"I need you to scrutinize and very carefully examine the image, pay attention to the finest details, and make use of any relevant information or small abnormalities."

    prompt = f"""USER: <image>\n{SYST_PROMPT}\n
    1. Does this image look realistic, considering lighting, shadows, and reflections?
    2. Are there any subtle, unexpected patterns or behaviors in the image that might not be immediately noticeable?
    3. can you clearly see {class_name} in the image?
    4. (if the answer of the previous question is 'No' answer 'Nan' to this question) Considering specific details like {class_name} posture, head and body shape, number of legs and form and the surroundings view, does the image look normal?
    5. (if response to 3. is 'No' answer 'Nan' to this question) If {class_name} is present, does it exhibit realistic and natural behavior?
    6. (if response to 3. is 'No' answer 'Nan' to this question) Are there any other objects or elements in the image that might be mistakenly identified as {class_name}?
    7. (if response to 3. is 'No' answer 'Nan' to this question) Does the representation of {class_name} in the image maintain anatomical accuracy (e.g., correct number of legs, tail, head)?
    8. Do the colors in the image accurately match the description in the PROMPT {description}, including subtle variations and shades?
    9. Can you identify any deviations or abnormalities in the image that might be less obvious but still significant?
    10. Given PROMPT: {description}. Does the image respect fully the prompt description?

    For each question, answer 'yes' or 'no' or 'nan' accordingly. If there are discrepancies, provide a brief description of the issues.\n\nASSISTANT: \n"""
    
    highest_score_indices = []
    
    for i in range(10):
        inputs = processor_llava(prompt, extracted_imgs[i], return_tensors='pt').to(0)
        output = model_llava.generate(**inputs, max_new_tokens=512*2)
        response_ = processor_llava.decode(output[0][2:], skip_special_tokens=False)
        answers_section_ = response_.split('ASSISTANT:')[1].split('</s>')[0].strip()
        answers = [ans.strip().split(' ')[1].lower() for ans in answers_section_.split('\n') if ans.strip()]  # Extracting individual answers
        
        # Calculate scores for each question
        scores = [1 if i == 1 or (i in [3, 4, 5, 6, 7, 8, 10] and ans == 'yes') or (i == 9 and ans == 'no') else -1 if i == 2 and ans == 'yes' else 0 for i, ans in enumerate(answers, start=1)]
        
        total_score = sum(scores)

        #print(f'image {i+1}: {total_score}') #if response with prompt: print(processor.decode(output[0][2:], skip_special_tokens=False))
        
        if not highest_score_indices or total_score > highest_score_indices[0][1]:
            highest_score_indices = [(i, total_score)]  
        elif total_score == highest_score_indices[0][1]:
            highest_score_indices.append((i, total_score)) 

        #print(scores)
        #print("-----------------------")

    indices = [index for index, _ in highest_score_indices]

    best_confidence_index = None
    best_confidence_value = 0.0

    model_yolo.set_classes([class_name])

    for i in indices:
        res = model_yolo.predict(extracted_imgs[i], conf=0.5)
        
        if res[0].boxes.xyxy.cpu().numpy().any():
            for j, prediction in enumerate(res):
                boxes = prediction.boxes.xyxy.cpu().numpy()
                confidences = prediction.boxes.conf.cpu().numpy()

                for box, confidence in zip(boxes, confidences):
                    if confidence > best_confidence_value:
                        best_confidence_value = confidence
                        best_confidence_index = i

        #else:
        #    print(f"0 detection for image at index {i}.")
        #    print(f"Image size: {extracted_imgs[i].size}")

    print("Image index with the best confidence value:")
    print(best_confidence_index)

    if best_confidence_index is not None:
        # Get the original filename without extension
        image_name = os.path.splitext(selected_image)[0]
        # Save the image with the original filename
        image_save_path = os.path.join(save_path, f"{image_name}.png")

        # Extract the directory path
        output_dir = os.path.dirname(image_save_path)

        # Check if the directory already exists
        os.makedirs(output_dir, exist_ok=True)  

        extracted_imgs[best_confidence_index].save(image_save_path)
        print(f"Image saved at: {image_save_path}")
    else:
        print("No image to save.")