## Text to images fine-tuning

Completed with the diffusers tutorial.  
Available here [stable](https://huggingface.co/docs/diffusers/v0.13.0/en/training/text2image#hardware-requirements-for-finetuning), or  [**latest (more details)**](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)


### Set-up
Following the guide, install locally the diffusers library from github (we do it within the srt2i env).  
```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements.txt
```

then 
```
accelerate config
```
>currently selecting only the basic options and *no* to any accelarator proposed.   

This config file will save in ```/home/<user>/.cache/huggingface/accelerate/default_config.yaml```


### 1. Training with hf dataset
Still following the diffusers tutorial, we [retrain on an existing dataset](https://huggingface.co/docs/diffusers/v0.13.0/en/training/text2image#finetuning-example).  
The script to retrain is [./diffusers/examples/text_to_image/0_myscript.sh](./0_myscript.sh).  
And uses the existing python script with various options.  

After training the model is run with the usual diffusers pipeline : [inference notebook](./inference_tests.ipynb)
```
from diffusers import StableDiffusionPipeline
import torch

model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("yoda-pokemon.png")
```

### 2. Training with custom dataset
Requires a dataset in the [correct format](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#image-captioning).  
Dataset requirement: 
 - folder with images 
 - metada.jsonl (in the same folder) with   
    ```{"file_name": "0001.png", "text": "This is an image caption/description"}```  
    ```{"file_name": "0002.png", "text": "This is an other image caption/description"} ```   
    ```... ```

See [inference nb](./inference_tests.ipynb) for a translation from a Kaggle dataset.  
**Implementation from generated dataset after the llava captioning step** : in [retrain notebook tutorial](../tutorial/retrain_sd_demo.ipynb)  

The script to retrain is slightly different this time [./1_custom_ds.sh](./1_custom_ds.sh).  
