# RUN THIS CODE USING THE FOLLOWING COMMANDE
# python llm/prompts_generator.py --model "TheBloke/Mistral-7B-Instruct-v0.2-AWQ" --class_list "llm/class_list.json" --output_prompts "generated_prompts_.txt"

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import json
import random
import re
import argparse

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
        device_map="cuda:0"
    )
    return model, tokenizer

def generate_prompts(class_list_path, output_path, model, tokenizer):
    # Load class list
    with open(class_list_path, 'r') as file:
        class_list = json.load(file)

    # Set seed for reproducibility
    seed = 42
    random.seed(seed)

    # Using the text streamer to stream output one token at a time
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # SYSTEM_PROMPT to instruct the model
    SYSTEM_PROMPT = "<s>[SYS] Generate concise prompts for a text-to-image diffusion generator using stable diffusion. Focus on the specified class give specific details of the scene and avoid additional information, notes, suggestions and discussions.[/SYS]"

    # Shuffle the classes to pick them randomly
    class_keys = list(class_list.keys())
    random.shuffle(class_keys)

    with open(output_path, 'w') as output_file:
        for class_id in class_keys:
            class_name = class_list[class_id]

            # Combine SYSTEM_PROMPT and class-specific prompt
            prompt = f"{SYSTEM_PROMPT}<s>[INST] Give 5 distinct, simple and detailed scene prompts for a text-to-image diffusion generator using stable diffusion with a {class_name} [/INST]"

            tokens = tokenizer(
                prompt,
                return_tensors='pt'
            ).input_ids.cuda()
            
            generation_params = {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_new_tokens": 512,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts using a text-to-image model.")
    parser.add_argument("--model", default="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", help="Name or path of the model.")
    parser.add_argument("--class_list", required=True, help="Path to the class list JSON.")
    parser.add_argument("--output_prompts", required=True, help="Path to the output text file for prompts.")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    generate_prompts(args.class_list, args.output_prompts, model, tokenizer)