import json
from glob import glob 
from natsort import natsorted

imgs_paths = natsorted(glob("../../optimal_pairs4/*.png"))


def txt_to_json(input_file, output_file):

    with open(input_file, 'r') as file:
        with open(output_file, 'w') as json_file:
            for im, line in zip(imgs_paths, file):
                print(im, line)
                image_class, description = line.strip().split('; ', 1)
                image_name = im.split('/')[-1]

                data = {"file_name": image_name, "text": description}
                
                json.dump(data, json_file)
                json_file.write('\n')  # Add a newline between entries

txt_to_json('../../generated_prompts2.txt', '../../optimal_pairs4/metadata.jsonl')