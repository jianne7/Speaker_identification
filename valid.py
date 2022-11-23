import glob
import os
import numpy as np
import random

random.seed(373)

base_dir = '/home/ubuntu/speaker_recognition/ujinne/voice_filter/data/feature_v2/'
speakers_path = os.path.join(base_dir, 'train')
valid_path = os.path.join(base_dir, 'valid')

spk_ids = sorted([f.split('/')[-1] for f in glob.glob(speakers_path + '/*') if os.path.isdir(f)])
for spk_id in spk_ids:
    file_list = os.listdir(os.path.join(speakers_path, spk_id))
    n_select = int(len(file_list)*0.2)
    valid_list = [f for f in random.sample(file_list, n_select) if f.endswith('.pt')]
    valid_dir = os.path.join(valid_path, spk_id)
    os.makedirs(valid_dir, exist_ok=True)

    for fname in valid_list:
        valid_fname = os.path.join(valid_dir, fname)
        speaker_fname = os.path.join(speakers_path, spk_id, fname)
        print(speaker_fname + ' ---> ' + valid_fname )
        os.rename(speaker_fname, valid_fname)