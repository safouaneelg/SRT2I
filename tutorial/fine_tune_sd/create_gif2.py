from diffusers import StableDiffusionPipeline
from natsort import natsorted
from glob import glob
from PIL import Image
import torch
import sys
import argparse

parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("modelname", type=str)
# parser.add_argument("outpath", type=str)
args=parser.parse_args()
model_path = args.modelname

# model_path = "./sd_optim_pair3_lora"

SEED = 42
generator = torch.Generator(device="cuda").manual_seed(SEED)


file = open('../../valid_prompts.txt', 'r')
lines = file.readlines()
prompts = [line[:-2] for line in lines]
# print(prompts)


pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32)
pipe.load_lora_weights(model_path)
pipe.to("cuda")

# image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator, cross_attention_kwargs={"scale":0.5}).images[0]
# image.save("create_gif0.png")

for i, prompt in enumerate(prompts):
    for scale in range(10):
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        images = pipe(prompt, cross_attention_kwargs={"scale":scale / 10}, generator=generator, num_images_per_prompt=5).images 
        for j, im in enumerate(images): 
            im.save(f"./gifs/imgs3/prompt{i}_{scale}_{j}.png")


    for k in range(5):
        frames = []
        for img in natsorted(glob(f"./gifs/imgs3/prompt{i}*{k}.png")):
            frames.append(Image.open(img))
        frames[0].save(f"./gifs/prompt{i}_{k}.gif", format="GIF", append_images=frames[1:], save_all=True, duration=500, loop=0)
