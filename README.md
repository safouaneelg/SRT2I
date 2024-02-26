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

2. Create a conda  environment (optional but recommended) with Python version >= 3.10
```bash
conda create --name srt2i python=3.10 -y
conda activate srt2i
pip install -r requirements.txt
```

### USAGE
**Step-by-step self-rewarding**

1. First step is to generate the prompts for generative text-to-image diffusion model. This could be achieved using the following command:

```python prompts_generator.py --model "TheBloke/Mistral-7B-Instruct-v0.2-AWQ" --class_list "llm/class_list.json" --output_prompts "generated_prompts.txt"```

The default parameters are `llm/class_list.json` (containing 80 classes from COCO datasets) and `generated_prompts.txt`. A total of 80*5=400 prompts are generated. In case you generate new prompts for other classes please change with the appropriate names.

2. The subsequent step is the generation of  images from prompts, which can be done by using diffusion model

## Licence

This code belongs to TOELT LLC and is open for research and development purposes only. No commercial use of this software is permitted.
For additional information, contact: safouane.elghazouali@toelt.ai