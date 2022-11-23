import os
import glob
import argparse
import shutil
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.transforms as T
import librosa

import numpy as np
import pandas as pd

# MelSpectrogram = T.MelSpectrogram(sample_rate=16000,
#                                   n_fft=512,
#                                   win_length=400,
#                                   hop_length = 160,
#                                   n_mels = 40)
    
MelSpectrogram = T.MelSpectrogram(sample_rate=16000,
                                  n_fft=512,
                                  win_length=512,
                                  hop_length = 128,
                                  n_mels = 80)

AmplitudeToDB = T.AmplitudeToDB()

def wav2melspectrogram(filepath: str, mode):
    wav, sample_rate = torchaudio.load(filepath)
    assert sample_rate == 16000, "Sampling Rate != 16000"

    if mode == 'train':
        # trim wav
        trimmed_wav, _ = librosa.effects.trim(np.array(wav))
        mel_specgram = MelSpectrogram(torch.from_numpy(trimmed_wav))
    else:
        mel_specgram = MelSpectrogram(wav)

    mel_specgram = AmplitudeToDB(mel_specgram)
    mel_specgram = mel_specgram.squeeze(0).transpose(0, 1).contiguous()

    return mel_specgram


def preprocess_data(dirpath: str, mode: str, savepath: str):
    print(f"Preprocessing {mode} data....")

    if mode == "train":
        spk_ids = sorted([f.split('/')[-1] for f in glob.glob(dirpath + '/*') if os.path.isdir(f)])

        for spk_id in tqdm(spk_ids):
            utts = glob.glob(f'{dirpath}/{spk_id}/*.wav')

            save_dir = os.path.join(savepath, spk_id)
            os.makedirs(save_dir, exist_ok=True)

            for utt in utts:
                mel_spec = wav2melspectrogram(utt, mode)
                if len(mel_spec) < 180:
                    pass
                else:
                    utt_name = utt.split('/')[-1].replace(".wav", ".pt")
                    torch.save(mel_spec, os.path.join(save_dir,utt_name))
 
    else:
        utts = glob.glob(f'{dirpath}/*.wav')
        os.makedirs(savepath, exist_ok=True)
        for utt in tqdm(utts):
            mel_spec = wav2melspectrogram(utt, mode)
            utt_name = utt.split('/')[-1].replace(".wav", ".pt")
            torch.save(mel_spec, os.path.join(savepath, utt_name))
    
    print("Preprocess Finished!!")


def unknown(dirpath: str):
    spk_ids = sorted([f.split('/')[-1] for f in glob.glob(dirpath + '/*') if os.path.isdir(f)])
    unk_path = os.path.join(dirpath, 'unknown')
    os.makedirs(unk_path, exist_ok=True)

    for spk_id in tqdm(spk_ids):
        file_list = os.listdir(os.path.join(dirpath,spk_id))
        # if len(file_list) < 15:
        if len(file_list) < 20:
            for f in file_list:
                shutil.move(os.path.join(dirpath, spk_id, f), unk_path)
            os.rmdir(os.path.join(dirpath, spk_id)) # 해당 디렉토리 삭제
        else:
            pass
    
    print("Move Unknown Folder!!")


# def train_meta(dirpath: str, savepath: str):
#     spk_ids = sorted([f.split('/')[-1] for f in glob.glob(dirpath + '/*') if os.path.isdir(f)])
#     spk_ids.append('unknown')
#     train_meta = pd.DataFrame({"id":spk_ids})
#     train_meta.to_csv(savepath+"train_meta.csv", index=False)
#     print("Saved train meta data!!")


def get_args():
    parser = argparse.ArgumentParser(description="Preprocessing the audio files from datasets.")
    parser.add_argument("--data_path", type= str, required=True, default='/data/', help="Dataset root path")
    parser.add_argument("--save_path", type= str, required=True, default='/data/feature/', help="Preprocessed data path")
    parser.add_argument("--mode", type= str, required=True, help="train or test mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    data_path = args.data_path
    save_path = args.save_path
    mode = args.mode

    data_dir = os.path.join(data_path, mode)
    print(f"data_dir:{data_dir}")
    assert os.path.exists(data_path)
    data_out_dir = os.path.join(save_path, mode)
    print(f"data_out_dir:{data_out_dir}")
    os.makedirs(save_path, exist_ok=True)
    assert os.path.exists(save_path)

    # Preprocess data
    preprocess_data(data_dir, mode, data_out_dir)
    # unknown
    if mode == "train":
        unknown(data_out_dir)

