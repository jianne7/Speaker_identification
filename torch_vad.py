import os
import glob
import argparse
from tqdm import tqdm

import librosa
import torch
import torchaudio
import numpy as np
from typing import Optional, Union
from pathlib import Path
from scipy.io.wavfile import write

sampling_rate = 16000

def trim_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
    # print(type(wav))

    # trim audio
    wav = torchaudio.functional.vad(torch.from_numpy(wav), source_sr)

    # # Apply the preprocessing: normalize volume and shorten long silences
    # wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    return np.array(wav)

def write2wav(wav: np.ndarray, savepath):
    sr = 16000
    write(savepath, sr, wav)

def main(args):

    data_path = os.path.join(args.data_dir, args.mode)
    save_path = os.path.join(args.save_dir, args.mode)

    if args.mode == "train":
        spk_ids = sorted([f.split('/')[-1] for f in glob.glob(data_path + '/*') if os.path.isdir(f)])

        for spk_id in tqdm(spk_ids):
            utts = glob.glob(f'{data_path}/{spk_id}/*.wav')

            save_dir = os.path.join(save_path, spk_id)
            os.makedirs(save_dir, exist_ok=True)

            for utt in utts:
                wav = trim_wav(utt)
                file_path = os.path.join(save_dir, utt.split('/')[-1])
                write2wav(wav, file_path)
     else:
        utts = glob.glob(f'{data_path}/*.wav')
        os.makedirs(save_path, exist_ok=True)

        for utt in tqdm(utts):
            wav = trim_wav(utt)
                file_path = os.path.join(save_path, utt.split('/')[-1])
                write2wav(wav, file_path)


def get_args():
    parser = argparse.ArgumentParser(description="Preprocessing the audio files from datasets.")
    parser.add_argument("--data_dir", type=str, required=True, default='/data/', help="Dataset root path")
    parser.add_argument("--save_dir", type=str, required=True, default='/data/vad/', help="Trimmed file save path")
    parser.add_argument("--mode", type=str, required=True, help="train or test mode")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)


