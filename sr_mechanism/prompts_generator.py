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
    SYSTEM_PROMPT = "<s>[SYS]in up to 77 tokens Generate very detailed scene description prompts for a text-to-image stable diffusion generative model. Focus on the specified class, give specific details of the scene with very comprehensive scene context description, describe as maximum as possible the finest detail and avoid additional information, notes, suggestions and discussions. Start describing the scene without naming it. Per each description only write text no need for enumeration. Use this example as template: 3 giraffes, in serene pond, middle of open air, two giraffes standing nearest to the water and one sitting outside, 3 trees behind them, 12 birds 7 flying and 5 on branches. ... etc. ONLY USE consistent enumeration style: 1. 2. 3. ... etc. No other enumeration format like 1- 2- or 1) 2) 3) is allowed.[/SYS]"

    # Shuffle the classes to pick them randomly
    # random.shuffle(class_keys)

    with open(output_path, 'w') as output_file:
        for class_id in class_keys:
            class_name = class_list[class_id]

            # Combine SYSTEM_PROMPT and class-specific prompt
            prompt = f"{SYSTEM_PROMPT}<s>[INST] Give {prompts_number} distinct, very descriptive and very detailed prompts for a text-to-image diffusion model focusing mainly on: {class_name}. The {class_name} should be the main focus and element of the scene, very clear and easy to see and locate and shoud be in different numbers colors each time. The prompts should be exaustive in different cases, {class_name} numbers, {class_name} scenarios, {class_name} positions and orientations. Details of the background, the colors, the observed {class_name} or multiple {class_name}s in the scene should also be given in a very exhaustive manner. The more given description details of every part of the image scene the better.[/INST]"

            tokens = tokenizer(
                prompt,
                return_tensors='pt'
            ).input_ids.cuda()
            
            generation_params = {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_new_tokens": 1024*8,
                "repetition_penalty": 1.0
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
