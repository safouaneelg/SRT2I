# SELF-REWARDING TEXT-TO-IMAGE MODELS
# python sr_mechanism/selfrewarded_t2i_v0.1.py --image_folder '/home/safouane/Downloads/SRT2I/generative_images/' --prompt_file '/home/safouane/Downloads/SRT2I/generated_prompts.txt' --image_descriptions_folder '/home/safouane/Downloads/SRT2I/generative_images_descriptions/' --output_folder './optimal_pairs/'
# 
from ultralytics import YOLO
from yolo_filter import yolo_filtering
from sentence_evaluator import *
import os
from PIL import Image
import time
from prompts_generator import load_model
from llava_descriptor import extract_images_from_grid

def extract_top_images(images_content, top_indices):
    # Split the images_content into individual image descriptions
    image_descriptions = images_content.split('\n')
    
    # Extract and keep only the content of the top images
    top_images_content = []
    for index in top_indices:
        try:
            top_images_content.append(image_descriptions[index-1])
        except IndexError:
            print(f"IndexError: Index {index} out of range for image_descriptions. Total images: {len(image_descriptions)}")
    
    # Join the content back into a single string
    top_images_content_combined = '\n'.join(top_images_content)

    return top_images_content_combined

def keep_top_n_values(values, n=3):
    sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
    top_indices = sorted_indices[:n]
    top_values = [values[i] for i in top_indices]
    return top_indices, top_values

def get_index_from_filename(filename):
    return int(filename.split('_')[0])

def main(image_folder, prompt_file, image_descriptions_folder, yolo_model='yolov8l-world.pt', llm_model='TheBloke/Mistral-7B-Instruct-v0.2-AWQ', output_folder='./optimal_pairs/'):
    
    start_time = time.time() 
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    existing_output_files = [f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        # Check if the image has already been processed and saved
        if os.path.basename(img_file) in existing_output_files:
            print(f"Skipping image: {img_path} (already processed)")
            continue

        print(f"Processing image: {img_path}")

        image = Image.open(img_path)

        # index from the chosen image file
        indice = get_index_from_filename(os.path.basename(img_file))-1

        prompt_list = []

        with open(prompt_file, 'r') as file:
            for line in file:
                class_, prompt = map(str.strip, line.split(';', 1))
                
                # [CLASS] and [PROMPT] to the 2D list
                prompt_list.append([class_, prompt])

        if 0 <= indice < len(prompt_list):
            file_name = f"{indice + 1}_{prompt_list[indice][0]}.txt"
            
            # content of the specified file
            with open(os.path.join(image_descriptions_folder, file_name), 'r') as image_file:
                images_content = image_file.read()
        else:
            print(f"Index {indice} is out of range.")
            continue

        class_name = [prompt_list[indice][0]]
        model = YOLO(yolo_model)
        model.set_classes(class_name)
        
        result = yolo_filtering(image, class_name)

        # splitting images_content to lines
        lines = images_content.split("\n")
        
        # filtering using yolo_filter
        lines_filtered = [line for i, line in enumerate(lines, start=1) if i not in result]

        scores_list = calculate_cosine_similarity_scores(prompt_list[indice][1], lines_filtered)
        
        # extract image numbers from lines_filtered
        image_numbers = [int(line.split(":")[0].split()[-1]) for line in lines_filtered]
        
        top_ind = []
        
        top_indices, top_values = keep_top_n_values(scores_list, n=3)
        for index, value in zip(top_indices, top_values):
            #print(f"Image {image_numbers[index]} - Score: {value}")
            top_ind.append(image_numbers[index])

        top_images_content = extract_top_images(images_content, [index for index in top_ind])

        # Print the content of the top images
        # print(top_images_content)
        
        model, tokenizer = load_model(llm_model)

        prompt = ""
        prompt_template = f'''{prompt}'''

        tokens = tokenizer(
            prompt_template,
            return_tensors='pt'
        ).input_ids.cuda()

        # Same as the "self-rewarding language models" paper
        generation_params = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_new_tokens": 512/4,
            "repetition_penalty": 1.1
        }

        SYSTEM_PROMPT = "<s>[SYS]Be critical in you analysis and very short in your answer. Give importance to the realistic aspect of the image. Answer the question directly, justify your answer. do not give unnecessary comments or discussions. Your answer should contain a single Image choice and don't talk about the other images.[/SYS]"

        prompt = f"{SYSTEM_PROMPT}<s>[INST]which one of those diffusion-based generated images <<{top_images_content}>> is accurately respecting the Given PROMPT: '{prompt_list[indice][1]}'. Please pay attention to details in the description and prompt, analyze the common features, colors, backgrounds and select a single Image out of all image descriptions. Your only answer should be the Updated PROMPT based on your analysis and make it very concise and very descriptive to the selected image. Follow this template for your answer: 'Image X: PROMPT.' where X is the number of the selected image and PROMPT is the updated PROMPT with detailed scene, colors, background, position, orientation.[/INST]"

        tokens = tokenizer(
            prompt,
            return_tensors='pt'
        ).input_ids.cuda()

        generation_output = model.generate(
            tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_params
        )
        
        generated_prompts = tokenizer.decode(generation_output[0])

        start_marker = '[/INST]'
        end_marker = ':'

        start_index = generated_prompts.find(start_marker) + len(start_marker)
        end_index = generated_prompts.find(end_marker, start_index)

        selected_image_description = generated_prompts[start_index:end_index].strip()
        
        images = extract_images_from_grid(image, 2, 5)

        image_nb = selected_image_description.split()[-1]

        img = images[int(image_nb)-1]

        os.makedirs(output_folder, exist_ok=True)

        # Save the image 
        output_image_path = os.path.join(output_folder, os.path.basename(img_path))
        img.save(output_image_path)

        print(f"Optimal pair saved at: {output_image_path}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time elapsed: {elapsed_time} seconds")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-rewarded Text-to-Image Generation")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images")
    parser.add_argument("--prompt_file", required=True, help="Path to the file containing prompts")
    parser.add_argument("--image_descriptions_folder", required=True, help="Path to the folder containing image descriptions")
    parser.add_argument("--yolo_model", default="yolov8l-world.pt", help="Path to YOLO model file")
    parser.add_argument("--llm_model", default="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", help="Path to LLM model file")
    parser.add_argument("--output_folder", default="./optimal_pairs/", help="Path to the folder for saving optimal pairs")

    args = parser.parse_args()

    main(args.image_folder, args.prompt_file, args.image_descriptions_folder, args.yolo_model, args.llm_model, args.output_folder)
