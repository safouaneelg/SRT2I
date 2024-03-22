from diffusers import StableDiffusionPipeline
import torch

#GOAL : loop over inference steps, guidance scale, lora scale, and prompts ( and seed ? )

inf_steps = [i for i in range(35, 81, 5)]
guidances = [i/2 for i in range(14, 24, 1)]
lora_scales = [i/10 for i in range(11)]
prompts = []

print(inf_steps, guidances, lora_scales)


## Normal 


prompt="Two elephants standing side by side in a lush green forest, one with its trunk raised high, \
    the other resting its massive body against a large tree. \
    Background: dense foliage and sunlight filtering through the leaves, creating dappled patterns on the ground." 
SEED = 42

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
model_path = "/home/arnaud/projects/srt2i/tutorial/fine_tune_sd/sdoptimpair4lora200"
pipe.load_lora_weights(model_path)
pipe.to("cuda")

# v0 
# for scale in lora_scales:
#     generator = torch.Generator(device="cuda").manual_seed(SEED)
#     images = pipe(prompt, num_inference_steps=65, guidance_scale=8, generator=generator, cross_attention_kwargs={"scale":scale}, num_images_per_prompt = 1).images
#     for i, im in enumerate(images):
#         im.save(f"sd21_lorascale_single/baseprompt{65}_{8.0}_{scale}_{i}.png")

# v1
for scale in lora_scales:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    image = pipe(prompt, num_inference_steps=65, guidance_scale=8, generator=generator, cross_attention_kwargs={"scale":scale}, num_images_per_prompt = 1).images[0]
    for i, im in enumerate(images):
        im.save(f"sd21_lorascale_single/baseprompt{65}_{8.0}_{scale}_{i}.png")
