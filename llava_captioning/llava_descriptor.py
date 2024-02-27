import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import time

torch.cuda.empty_cache()

# Model
def llava_loader(model_id="llava-hf/llava-1.5-7b-hf"):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

# Image separator
def extract_images_from_grid(grid, rows, cols):
    images = []
    for i in range(rows):
        for j in range(cols):
            box = (j * grid.width // cols, i * grid.height // rows, (j + 1) * grid.width // cols,
                   (i + 1) * grid.height // rows)
            images.append(grid.crop(box))
    return images

# sorting the images by their number
def numerical_order(filename):
    return [int(part) if part.isdigit() else part for part in filename.split('_')]

def main(args):
    # loading LlaVa model
    model, processor = llava_loader(args.llava_model)
    
    #prompt = "USER: <image>\nIn no more that a single unique detailed paragraph: Describe the scene depicted in the image generated by the stable diffusion model. Pay attention to details such as the main subject, colors, textures, and any prominent background elements. Include the numbers and colors of elements in your description such as 'red cat' 'orange tulips' 'white horse' 'green grace' 'grey elephant' 'black dog' 'clean water' '5 trees' 'white bear' 'open field' ...etc. Provide a vivid and comprehensive description as if narrating the visual content of the image. Additionally, please mention if there are any unusual elements or abnormalities present. Finally, provide your assessment of the image's realism and comment on whether it appears to be a well-executed drawing, a bad drawing, or a piece of art. Avoid additional information, unnecessary comments, and discussions.\nASSISTANT: \n"
    
    # prompt = "USER: <image>\nIn no more that a single unique detailed paragraph generate very detailed scene description of the image. Pay attention to details such as the main subject, colors, textures, and any prominent background elements. Include the numbers and colors of elements in your description such as 'red cat' 'orange tulips' 'white horse' 'green grace' 'grey elephant' 'black dog' 'clean water' '5 trees' 'white bear' 'open field' ...etc. Provide a vivid and comprehensive description as if narrating the visual content of the image. Additionally, please mention if there are any unusual elements or abnormalities present. Finally, provide your assessment of the image's realism and comment on whether it appears to be a well-executed drawing, a bad drawing, or a piece of art. Avoid additional information, unnecessary comments, and discussions.\nASSISTANT: \n"
    
    prompt = "USER: <image>\nGenerate very detailed scene description of the image. describe in very detailed sentences the finest possible details of the image. Comment on the image main element and center focus as well as the background. You need also to mention is the image is realistic or drawing or art. You need to mention also the abnormalities observed in the image and unfamiliar aspects. Don't forget to cite the number of elements you see, the colors and anything worth mentioning. Use one of the following templates: TEMPLATE 1: 'This image is most likely a realistic one. Polar bear mother and her two cubs playing on a chunk of ice in the Arctic sea. The mother bear's thick, white fur is covered in snow and ice, and her sharp claws dig into the ice. The cubs' fur is soft and white, and they playfully wrestle with each other. In the background, a family of walruses lounges on the ice. The abnormality is that polar bear mother eyes look wrongly positioned'. TEMPLATE 2: 'Red cat sitting on a sunlit windowsill looking out at a lush green garden. background: tulips in full bloom, bees hovering around, a blue bird perched on a garden gnome. The image is most likely an art. Some abnormalities have been notices such as...'. Start directly describing the image and avoid discussion, notes or suggestions. \nASSISTANT: \n"

    image_folder = args.image_folder
    output_folder = args.output_describ_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
                         key=numerical_order)
    
    start_time = time.time() 

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)

        image = Image.open(image_path)
        print(f"\n{image_file}\#-------------------/#/\n")
        plt.imshow(image)
        plt.show()

        ################### EXTRACT SINGLE IMAGE (BECAUSE BELOW IMAGES ARE GROUPED 10 IMAGES)
        extracted_imgs = extract_images_from_grid(image, 2, 5)

        ################### PERFORM IMAGE TO TEXT
        assistant_descriptions = []

        for j in range(10):
            print(f"\nimage {j + 1}\#\-------------------/#/\n")
            inputs = processor(prompt, extracted_imgs[j], return_tensors='pt').to(0, torch.float16)

            plt.imshow(extracted_imgs[j])
            plt.show()

            output = model.generate(**inputs, max_new_tokens=512 * 8, do_sample=False)
            description = processor.decode(output[0][2:], skip_special_tokens=True)

            # Remove everything before and including "ASSISTANT:"
            start_index = description.find("ASSISTANT:")
            if start_index != -1:
                description = description[start_index + len("ASSISTANT:"):].strip()

            assistant_descriptions.append(description)

        ################### SAVE TO TEXT FILE
        output_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")

        with open(output_file, 'w') as f:
            f.write('\n'.join([f'image {i + 1}: {desc}' for i, desc in enumerate(assistant_descriptions)]))

        print(f"Descriptions for {image_file} saved to {output_file}")
        
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\nTotal time elapsed: {elapsed_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LlaVa Image Descriptor')

    parser.add_argument('--llava-model', type=str, default="llava-hf/llava-1.5-7b-hf",
                        help='Name or path of the LlaVa model. Default is "llava-hf/llava-1.5-7b-hf".')

    parser.add_argument('--output_describ_folder', type=str, default='generative_images_descriptions/',
                        help='Path to the output folder for saving descriptions. Default is "generative_images_descriptions/".')

    parser.add_argument('image_folder', type=str, help='Path to the folder containing images.')

    args, unknown = parser.parse_known_args()
    main(args)
