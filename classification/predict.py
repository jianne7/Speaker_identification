import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob
import random
import numpy as np
import pandas as pd  # pandas v1.3.5
from tqdm import tqdm 
from collections import OrderedDict

from model.lstm_model import LSTM
# from model.transformer_model import SpeakerTransformer
# from model.conformer_model import SpeakerConformer


torch.cuda.empty_cache()


def index2char(speaker_id: list):
    txt2id = {f'{txt}':i for i, txt in enumerate(speaker_id)}
    # txt2id['unknown'] = len(txt2id)+1
    # id2txt = {value:key for key, value in txt2id.items()}
    id2txt = {int(value):key for key, value in txt2id.items()}

    return txt2id, id2txt


def init_model(model_path, class_num, device):
    input_size = 80
    hidden_size = 768
    num_layers = 3
    output_size = class_num

    model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)
    # model = SpeakerTransformer(device, output_size).to(device)
    # model = SpeakerConformer(device, output_size).to(device)

    if isinstance(model, nn.DataParallel):  # GPU 병렬사용 적용
        model.load_state_dict(torch.load(model_path))
    else: # GPU 병렬사용을 안할 경우
        state_dict = torch.load(model_path) 
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`  ## module 키 제거
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load(model_path), strict=False)
    
    return model


def prediction(dirpath: str, model, device):
    model.eval()
    with torch.no_grad():
        mel_specgram_tensor = torch.load(dirpath) # (1, T, 40) utterance
        mel_spec = mel_specgram_tensor.squeeze(0).to(device)
        # sampling_frame_length = 160 # time
        len_time = 160
        
        # if len_time < sampling_frame_length:
        if mel_spec.size(0) < len_time:
            padder = torch.zeros(len_time-mel_spec.size(0), mel_spec.size(1)).to(device)
            mel_spec = torch.cat([mel_spec, padder], dim=0)
        #     len_time = mel_spec.size(0)

        output = model.predict(mel_spec)

    return output


def classification(utt_list: list, id2txt: dict, model, device, thres=0.5):
# def classification(utt_list: list, id2txt: dict, model, device):
    spk_id = []
    for utt in tqdm(utt_list):
        model.eval()
        with torch.no_grad():
            output = prediction(utt, model, device)
            spk_id.append(predict(output, topk=(1,), thres=thres))
        torch.cuda.empty_cache()

    return spk_id
    

def predict(output, topk=(1,), thres=0.5):
# def predict(output, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = 1

        values, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # Getting the most probable speaker_id(str)
        prediction = id2txt.get(int(pred[0][0]))
        # Custom
        if torch.max(values.softmax(-1)).item() < thres:
            prediction = 'unknown'

    return prediction



if __name__ == '__main__':
    model_path = "/home/ubuntu/speaker_recognition/ujinne/voice_filter/res_se/vL+L0.1_2/20298_train_lstm.pt"
    device = torch.device("cuda")
    train_dir = '/home/ubuntu/speaker_recognition/ujinne/voice_filter/data/feature/train'
    test_dir = '/home/ubuntu/speaker_recognition/ujinne/voice_filter/data/feature/test'
    
    spk_ids = sorted([f.split('/')[-1] for f in glob.glob(train_dir + '/*') if os.path.isdir(f)])
    _, id2txt = index2char(spk_ids)
    class_num = len(id2txt)

    model = init_model(model_path, class_num, device)

    utt_list = sorted([f for f in glob.glob(test_dir + '/*.pt') if os.path.isfile(f)])
    spk_id = classification(utt_list, id2txt, model, device, thres=0.8)
    print("---------------Classification Finished!!!-----------------")
    recording = [f.split('/')[-1].replace(".pt", ".wav") for f in utt_list]

    submission = pd.DataFrame({"recording" : recording, "voice_id" : spk_id})
    submission = submission.sort_values(by='recording')
    submission.to_csv("submission.csv", index=False)