# RUN THIS CODE USING THE FOLLOWING COMMANDE
# python llm/prompts_generator.py --model "TheBloke/Mistral-7B-Instruct-v0.2-AWQ" --class_list "llm/class_list.json" --output_prompts "generated_prompts.txt" --prompts_number 30 --class_ids 15,16,17,20,21

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import json
import random
import re
import argparse
import time
import torch

torch.cuda.empty_cache()

def load_model(model_name_or_path="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"):
    """Load the tokenizer and the model

    Args:
        model_name_or_path (string)
        takes as input huggingeface pre-trained instruct models:
        'togethercomputer/Llama-2-7B-32K-Instruct' 'TheBloke/Llama-2-7B-32K-Instruct-AWQ' 'dfurman/Llama-2-13B-Instruct-v0.2' 'TheBloke/Llama-2-7B-32K-Instruct-GGUF' 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ' 'tiiuae/falcon-7b-instruct' ...etc

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        device_map="cuda"
    )
    return model, tokenizer

def generate_prompts(class_list_path, output_path, model, tokenizer, prompts_number=10, class_ids=None):
    start_time = time.time()
    # Load class list
    with open(class_list_path, 'r') as file:
        class_list = json.load(file)

    if class_ids:
        class_ids = [int(id) for id in class_ids.split(',')]
        class_keys = [str(class_id) for class_id in class_ids if str(class_id) in class_list]
    else:
        class_keys = list(class_list.keys())

        
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)

    # Using the text streamer to stream output one token at a time
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # SYSTEM_PROMPT to instruct the model
    SYSTEM_PROMPT = """
    <s>[SYS]Please generate short but very detailed scene description prompts for a text-to-image stable diffusion model. The scene should be realistic with natural lighting. Focus on the specified class, give specific details of the scene with very comprehensive context description. Describe as maximum as possible the finest details and avoid additional information, notes, suggestions and discussions. Start describing the scene without naming it. Per each description only write text no need for enumeration. ONLY USE consistent enumeration style: 1. 2. 3. ... etc. No other enumeration format like 1- 2- or 1) 2) 3) is allowed.    
    The prompts should be in key worlds only, not text, and should the cameras brand usually used to capture high resolution photos as well as the description of the scene. Don't repeat scenes and propose new scenes and photo descriptions.[/SYS]
    """

    # Shuffle the classes to pick them randomly
    # random.shuffle(class_keys)

    with open(output_path, 'w') as output_file:
        for class_id in class_keys:
            class_name = class_list[class_id]

            # Combine SYSTEM_PROMPT and class-specific prompt
            prompt = f"""
            {SYSTEM_PROMPT}<s>[INST] Give {prompts_number} distinct prompts for a text-to-image model focusing mainly on: {class_name}. The {class_name} should be the main focus and element of the scene, very clear and easy to see and locate. Additionally, {class_name} should be in different numbers, colors, positions (like sleeping, resting, eating, looking, walking) at each prompt. The prompts should be in very different cases and real scenarios. The prompts should be in key worlds only, not text, and should the cameras brand usually used to capture high resolution photos as well as the description of the scene. Different {class_name} numbers, {class_name} scenarios, {class_name} positions and orientations should be given.
            Generate prompts detailed and similar to those templates:
            - 'Wide shot of two walking majestic elephants, One elephant up the hill, Other small elephant following him. background: far mountains are visible. Photo, Photograph, Natural lighting, Canon EOS 5D Mark IV, RAW photo.\n'
            - 'Photo of one tall giraffe, reaching for a tree branch, Giraffe in water with reflection, Background: tree with a few leaves, and the sky can be seen in the distance, sharp, photography, Nikon D850, 50mm, f/2.8, natural lighting\n'
            - 'Nature photo capture of one tall giraffe, reaching for a tree branch, Giraffe in water with reflection, Background: tree with a few leaves, and the sky can be seen in the distance, sharp, photography, Nikon D850, 50mm, f/2.8, natural lighting\n'
            - 'Three large brown bears walking across a fallen tree, with a small white dog sitting on the log in front. Background: walruses are visible and family of walruses lounges on the ice, sharp, photography, Nikon D850, 50mm, f/2.8, natural lighting\n'
            start your prompt with 'Wide photo of' or 'Photo of' or 'HD picture of' or 'Nature capture of' etc. Then proceed by describing the position, orientation, action of the {class_name} like 'eating from' or 'playing with' or 'bathing in' or 'sleeping' or 'walking close to' or 'looking up' 'looking forward' or 'shout' or 'openning his mouth' or 'running toward' or 'fighting against' or 'drinking from' etc. Following this describe the image view (morning or afternoon or night or sunrise or sunset or cloudy or rainy or snowy or midst of clouds etc) and scene details such as 'close to tree' 'in desert' or 'in mud' or 'in forest' or 'in Savanna' or 'in water' or 'in river' or 'thick mist' etc. Then describe background such as 'mountain hills' or 'large trees' or 'sunset' or 'herd of wild animals' or 'sea' or 'lake' or 'blue sky' or 'cloudy sky' or 'large trees' or 'Rainforest' etc. Finish the prompt by 'sharp, photography, (camera model such as Canon EOS M6 or Leica SL2 or Hasselblad X1D II 50C or Fujifilm X-T2 or Nikon D850, 50mm, f/2.8 etc.), natural lighting, RAW photo.[/INST]
            """

            tokens = tokenizer(
                prompt,
                return_tensors='pt'
            ).input_ids.cuda()
            
            generation_params = {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_new_tokens": 1024*12,
                "repetition_penalty": 1.1
            }

            generation_output = model.generate(
                tokens,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                **generation_params
            )

            # Decode and write the generated text with class name to the file
            generated_prompts = tokenizer.decode(generation_output[0]).split('\n')
            for i, prompt in enumerate(generated_prompts):
                ####### REFORMATING THE TEXT #########
                if i == 0:
                    # Remove the entire 'prompt' from the text
                    prompt = prompt.split('[/INST] ')[-1].replace('</s>', '')
                else:
                    prompt = prompt.replace('<s>', '').replace('</s>', '')

                # Remove enumerations '1.', '2.', '3.', '4.', '5.'
                prompt = re.sub(r'\d+\.', '', prompt)

                if prompt:
                    output_file.write(f"{class_name}; {prompt}\n")
                    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts using a text-to-image model.")
    parser.add_argument("--model", default="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", help="Name or path of the model.")
    parser.add_argument("--class_list", required=True, help="Path to the class list JSON.")
    parser.add_argument("--output_prompts", required=True, help="Path to the output text file for prompts.")
    parser.add_argument("--prompts_number", type=int, default=10, help="Number of prompts to generate for each class.")
    parser.add_argument("--class_ids", type=str, help="List of class IDs to generate prompts for. IDs should be separated by commas.")
    args = parser.parse_args()    

    # Load class list
    with open(args.class_list, 'r') as file:
        class_list = json.load(file)

    # Display names of chosen classes
    if args.class_ids:
        class_ids = [int(class_id) for class_id in args.class_ids.split(',')]
        chosen_classes = [class_list[str(class_id)] for class_id in class_ids]
        print(f"Chosen classes: {', '.join(chosen_classes)}")

    model, tokenizer = load_model(args.model)
    generate_prompts(args.class_list, args.output_prompts, model, tokenizer, args.prompts_number, args.class_ids)
