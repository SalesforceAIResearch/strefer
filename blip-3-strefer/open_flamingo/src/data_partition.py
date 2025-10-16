import os
import json
from tqdm import tqdm
import random
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True)
parser.add_argument('--data_path', required=True)
args = parser.parse_args()

print(f"Data root is: {args.data_root}")
print(f"Data config path is: {args.data_path}")

with open(args.data_path, "r") as file:
    config = yaml.safe_load(file)

print(config)
print(config["data_path"])

print("Checking if data preprocessing was done...")
missing = False
for data_path in config["data_path"]:
    if not os.path.exists(data_path):
        missing = True
        break

if missing:
    print("Preprocessing data.json...")
    num_partitions = len(config["data_path"])

    print("Loading data.json...")
    with open('data/data.json', 'r') as f:
        data = json.load(f)
    print("Finished loading data.json!")
    
    assert sum(config["data_path"].values()) == len(data)
    
    random.shuffle(data)
    
    video_refer_unique_path_prefix = {'A2D-Sentences', 'MeViS', 'Refer-YouTube-VOS', 'VideoRefer-700K'}
    
    for i in range(len(data)):
        if data[i]['path'].split('/')[0] in video_refer_unique_path_prefix:
            data[i]['videorefer'] = True  # add the 'videorefer' key
        data[i]['path'] = os.path.join(args.data_root, data[i]['path'])  # turn path into full path
    
    for i in tqdm(range(len(data))):
        for t_idx in range(len(data[i]['conversations'])):
            for turn_idx in range(len(data[i]['conversations'])):
                # '<video>\n' to '<image>\n'
                data[i]['conversations'][turn_idx]['value'] = data[i]['conversations'][turn_idx]['value'].replace('<video>', '<image>')

    data_idx = 0
    for data_path in config["data_path"]:
        print(data_path, data_idx, data_idx+config["data_path"][data_path], len(data[data_idx:data_idx+config["data_path"][data_path]]))
        with open(data_path, 'w') as f:
            json.dump(data[data_idx:data_idx+config["data_path"][data_path]], f)
        data_idx += len(data[data_idx:data_idx+config["data_path"][data_path]])
        print('{} saved!'.format(data_path))
else:
    print("Nothing to do!")
    
        