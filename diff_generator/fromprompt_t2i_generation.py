import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import argparse
import time

def load_diffusion_model(diffusion_model_name):
    # Initialize the StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(diffusion_model_name, torch_dtype=torch.float16)
    pipe = pipe.to('cuda:2')
    print('Stable Diffusion Model successfully loaded')
    return pipe

def generate_images(prompts_file_path, output_folder, pipe, generator):
    # Load the generated prompts from the file
    with open(prompts_file_path, 'r') as prompts_file:
        prompts_lines = prompts_file.readlines()
    print(f'Prompt file {prompts_file_path} successfully loaded')

    # Seed for reproducibility
    generator.manual_seed(1024)

    print('Starting the generation of images ...')

    start_time = time.time()

    # Loop over the prompts and generate images
    for i, prompt_line in enumerate(prompts_lines):
        parts = prompt_line.strip().split(';')
        class_name = parts[0].strip().lower().replace(" ", "_")
        prompt = ';'.join(parts[1:]).strip()

        # Generate 10 images per prompt
        num_images = 10
        images = pipe([prompt] * num_images, generator=generator).images
        grid = image_grid(images, rows=2, cols=5)

        image_name = f"{i+1}_{class_name}.png"
        image_path = os.path.join(output_folder, image_name)
        grid.save(image_path)

        print(f"Generated images for prompt {i+1}: {image_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Images have been stored in {output_folder}')
    print(f'Total elapsed time: {elapsed_time:.2f} seconds')

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts using a text-to-image diffusion model.")
    parser.add_argument("--diffusion-model", default="stabilityai/stable-diffusion-2-1-base", help="Name or path of the diffusion model.")
    parser.add_argument("--output-folder", required=True, help="Path to the output folder for generated images.")
    parser.add_argument("--prompts", required=True, help="Path to the prompts file.")
    args = parser.parse_args()

    # Load the diffusion model
    pipe = load_diffusion_model(args.diffusion_model)

    # Create the output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Set seed for reproducibility
    generator = torch.Generator(device="cuda:2")

    # Generate images
    generate_images(args.prompts, args.output_folder, pipe, generator)
