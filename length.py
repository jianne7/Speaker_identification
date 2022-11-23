import os
import glob
import argparse
import shutil
from tqdm import tqdm
import librosa
import torch
import torchaudio
import torchaudio.transforms as T

import numpy as np
import pandas as pd


MelSpectrogram = T.MelSpectrogram(sample_rate=16000,
                                  n_fft=512,
                                  win_length=512,
                                  hop_length = 128,
                                  n_mels = 80)

AmplitudeToDB = T.AmplitudeToDB()

def wav2melspectrogram(filepath: str):
    samples, sample_rate = torchaudio.load(filepath)
    assert sample_rate == 16000, "Sampling Rate != 16000"

    # trim wav
    samples, _ = librosa.effects.trim(np.array(samples))
    # samples = torchaudio.functional.vad(samples, sample_rate)

    mel_specgram = MelSpectrogram(torch.from_numpy(samples))
    mel_specgram = AmplitudeToDB(mel_specgram)
    mel_specgram = mel_specgram.squeeze(0).transpose(0, 1).contiguous()

    return mel_specgram


def length(datadir):
    spk_ids = sorted([f.split('/')[-1] for f in glob.glob(datadir + '/*') if os.path.isdir(f)])
    speaker = []
    original = []
    total_cnt = []
    for spk_id in tqdm(spk_ids):
        utts = glob.glob(f'{datadir}/{spk_id}/*.wav')
        len_utts = len(utts)

        count = 0
        for utt in utts:
            mel_spec = wav2melspectrogram(utt)
            if len(mel_spec) < 180:
                count += 1
            else:
                pass
        
        if count > 0:
            speaker.append(spk_id)
            original.append(len_utts)
            total_cnt.append(count)
        
    return speaker, original, total_cnt

def length_file(datadir):
    spk_ids = sorted([f.split('/')[-1] for f in glob.glob(datadir + '/*') if os.path.isdir(f)])
    total = 0
    for spk_id in tqdm(spk_ids):
        utts = glob.glob(f'{datadir}/{spk_id}/*.pt')
        total += len(utts)

    return total

def unknown_len(dirpath: str):
    spk_ids = sorted([f.split('/')[-1] for f in glob.glob(dirpath + '/*') if os.path.isdir(f)])
    under_20=[]
    over_20=[]
    for spk_id in tqdm(spk_ids):
        utts = glob.glob(f'{dirpath}/{spk_id}/*.wav')
        len_utts = len(utts)
        utt_list = []
        for utt in utts:
            mel_spec = wav2melspectrogram(utt)
            if len(mel_spec) < 180:
                pass
            else:
                utt_list.append(utt)

        if len(utt_list) < 20:
            under_20.append(spk_id)
        else:
            over_20.append(spk_id)

    return under_20, over_20
            

if __name__ == "__main__":
    data_dir = '/home/ubuntu/speaker_recognition/data/train'
    # speaker, original, total_cnt = length(data_dir)
    # df = pd.DataFrame({"speakerID" : speaker,
    #                     "original_length" : original,
    #                     "count_num" : total_cnt})
    # df.to_csv('length.csv')
    # len_file = length_file(data_dir)
    # print(f'Total length : {len_file}')
    under, over = unknown_len(data_dir)
    print(f'Total under : {len(under)}')
    print(f'Total over : {len(over)}')

    

