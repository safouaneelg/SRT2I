# SRT2I: Self-rewarding generative text-to-image models

## INTRODUCTION

Self-rewarding mechanism
![selfrewarding](_repoimages_/T2I_selfrewarding_mechanism.gif)

## USAGE INSTRUCTION

###  INSTALLATION
1. Clone the repository and install dependencies using

```bash
git clone https://github.com/safouaneelg/SRT2I.git
```

2. Create a conda  environment (optional but recommended) from environment.yml
```bash
conda env create -f environment.yml
conda activate srt2i
```

### USAGE
**Step-by-step self-rewarding**

1. First step is to generate the prompts for generative text-to-image diffusion model. This could be achieved using the following command:

```bash
python llm/prompts_generator.py --model "TheBloke/Mistral-7B-Instruct-v0.2-AWQ" --class_list "llm/class_list.json" --output_prompts "generated_prompts.txt" --prompts_number 30 --class_ids 15,16,17,20,21
```

The default parameters are:
 - `"TheBloke/Mistral-7B-Instruct-v0.2-AWQ"` Mistral 7B quantized
 - `llm/class_list.json` (containing 80 classes from COCO datasets)
 - `generated_prompts.txt` A total of 80*5=400 prompts are generated and stored in this file txt format
 - `prompts_number` The default number of prompts per class is 30

Those are the class ids used in the paper: {15:cat}, {16:dog}, {17:horse}, {20:elephant}, {21:bear}
In case you generate new prompts for other classes please change with the appropriate ids in [class_list](llm/class_list.json).


2. The subsequent step is the generation of  images from prompts, which can be done by using stable diffusion model.
Run the following command to generate the images. The images are stacked by 10

```bash 
python diff_generator/fromprompt_t2i_generation.py --diffusion-model "stabilityai/stable-diffusion-2-1-base" --output-folder "generative_images/" --prompts "generated_prompts.txt"
```

The default parameters are:
 - `"stabilityai/stable-diffusion-2-1-base"` stable diffusion 2.1
 - `generative_images/` folder to store the generated images
 - `generated_prompts.txt` A total of 80*5=400 prompts are generated and stored in this file txt format
 - `prompts_number` The default number of prompts per class is 30

3. 

```bash 
python sr_mechanism/selfrewarded_t2i_v0.1.py --image_folder 'path/to/images/folder/' --prompt_file 'path/to/prompts_file.txt' --image_descriptions_folder 'path/to/llava/descriptions/' --yolo_model 'YOLO_WORLD_MODEL' --llm_model 'LLM_MODEL' --output_folder 'path/to/output/storage/folder/'
```

Parsers:
 - **image_folder**: path to the folder containing stacked 10 images (typically  `generative_images/` in this code)
 - **prompt_file**: path to the `.txt` file containing all the prompt (`generated_prompts.txt`)
 - **image_descriptions_folder**: path to the folder where `*.txt` llava captioning for the stacked 10 images has been saved (`./generative_images_descriptions/`)
 - **yolo_model**: Open Vocabulary YOLO model set by default to `'yolov8l-world.pt'` but can also be changed to `'yolov8m-world.pt'` or `'yolov8s-world.pt'`
 - **llm_model**: LLM model for self-judging set by default to `'TheBloke/Mistral-7B-Instruct-v0.2-AWQ'` 
 - **output_folder**: `'path/to/output/storage/folder/'` where optimal images to specific prompts in the **prompt_file** in the will be saved

## Licence

This code is open for research and development purposes only. No commercial use of this software is permitted.
For additional information, contact: safouane.elghazouali@toelt.ai.