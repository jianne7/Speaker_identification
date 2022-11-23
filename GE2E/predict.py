import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio
# import torchaudio.transforms as AT

import os
import glob
import numpy as np
import pandas as pd  # pandas v1.3.5
from tqdm import tqdm 
import pickle
from collections import OrderedDict

torch.cuda.empty_cache()


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    return data

def save_pickle(save_path, data):
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def get_dvec(dirpath: str, spk_id: str, model, device):
    model.eval()
    with torch.no_grad():
        utts = glob.glob(f'{dirpath}/{spk_id}/*.pt')

        dvecs = []      
        for utt in utts:
            mel_specgram_tensor = torch.load(utt) # (1, T, 40)
            mel_spec = mel_specgram_tensor.squeeze(0).to(device)
            len_time = 160 # time
            # print(f'tensor_size1:{utt_tensor.size(1)}')
            # print(f'time:{len_time}')

            if mel_spec.size(0) < len_time:
                # print(f'prev_tensor:{utt_tensor.size(0)}')
                padder = torch.zeros(len_time-mel_spec.size(0), mel_spec.size(1)).to(device)
                mel_spec = torch.cat([mel_spec, padder], dim=0)

            dvec = model(mel_spec)
            dvecs.append(dvec)
        
        dvecs = torch.stack(dvecs)  # (M, 256)
        centroid = torch.mean(dvecs, dim=0).unsqueeze(0) # (1, 256)
        # print(f'dvec:{dvecs.shape}')
        # print(f'cent:{centroid.shape}')

        max_dist = max_distance(dvecs, centroid)
    
    return centroid, dvecs, max_dist


def max_distance(dvecs:list, centroid):
    dists = []
    for dvec in dvecs:
        dvec = dvec.unsqueeze(0)
        dist = torch.cdist(centroid, dvec, p=2)
        dists.append(dist.squeeze(0))
    max_dist = max(dists)

    return max_dist


def make_classes(train_dir, model, device):
    spk_ids = sorted([f.split('/')[-1] for f in glob.glob(train_dir + '/*') if os.path.isdir(f)])

    classes = []
    d_vectors = []
    for spk_id in tqdm(spk_ids): 
        # avg_dvec, thres = get_dvec(data_dir, spk_id, model, device, model_path)
        cent, dvecs, max_dist = get_dvec(train_dir, spk_id, model, device)
        classes.append({"spk_id" : spk_id,
                        "centroid" : cent,
                        "max_dist" : max_dist})

        d_vectors.append({"spk_id" : spk_id,
                          "d_vectors" : dvecs})

        del cent
        del dvecs
        del max_dist
        torch.cuda.empty_cache()

    save_pickle("classes.pickle", classes)
    save_pickle("d_vectors.pickle", d_vectors)  # for using visualization
    print("---------------Saved Classes data!!!-----------------")

    return classes


def get_test_dvec(dirpath: str, model, device):
    model.eval()
    with torch.no_grad():
        mel_specgram_tensor = torch.load(dirpath) # (1, T, 40) utterance
        mel_spec = mel_specgram_tensor.squeeze(0).to(device)
        len_time = 160 # time
        
        if mel_spec.size(0) < len_time:
            padder = torch.zeros(len_time-mel_spec.size(0), mel_spec.size(1)).to(device)
            mel_spec = torch.cat([mel_spec, padder], dim=0)

        dvec = model(mel_spec)

    return dvec


def classification(utt_list: list, classes, model, device):
    spk_id = []
    for utt in tqdm(utt_list):
        dvec = get_test_dvec(utt, model, device)
        distance = []
        for i in range(len(classes)):
            dist = torch.cdist(classes[i]['centroid'], dvec.unsqueeze(0), p=2) # euclidean
            distance.append(dist)

        min_distance = min(distance) # find the shortest distance
        index = distance.index(min_distance) # the shortest distance's index

        print(f'min_distance : {min_distance}')
        print(f'max_distance : {classes[index]["max_dist"]}')

        if min_distance <= classes[index]['max_dist']:
            spk_id.append(classes[index]['spk_id'])
            print(f"spk_id: {classes[index]['spk_id']}")
        else:
            spk_id.append('unknown')
            print("spk_id: unknown")
        # break
        del dvec
        torch.cuda.empty_cache()

    print("---------------Classification Finished!!!-----------------")

    return spk_id


class LSTM(nn.Module):
    def __init__(self,input_size=40, hidden_size=768, num_layers=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = nn.Linear(self.hidden_size, 256)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        
    def forward(self, x):
        # print(f'x_shape : {x.shape}') 
        x = x.transpose(0, 1) # ex) (333, 40) -> (40, 333)
        x = x.unfold(1, 160, 80) # (num_mels, T', window)  ex) 40, 3, 160
        x = x.permute(1, 2, 0) # (T', window, num_mels))   ex) 3, 160, 40
        # x = x.squeeze(0)
        # print(f'x_shape : {x.shape}')
#         self.lstm.flatten_parameters()
        y,_ = self.lstm(x)
        y = y[:, -1, :] # (T', lstm_hidden), use last frame only
        # print(f'y[:, -1, :]_shape : {y.shape}')
#         y = torch.mean(y, dim=1)
        y = self.layers(y) # (BS, T, emb_dim)
        y = y / torch.norm(y, p=2, dim=1, keepdim=True) # (BS, T, emb_dim)
        # y = y.sum(1) / y.size(1) # (emb_dim), average pooling over time frames
        # print(f'y / torch.norm_shape : {y.shape}')
        y = torch.mean(y, dim=0)
        # print(f'y_shape : {y.shape}')
        
        return y


def init_model(model_path, device):
    input_size = 40
    hidden_size = 768
    num_layers = 3

    model = LSTM(input_size, hidden_size, num_layers).to(device)

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


if __name__ == '__main__':
    model_path = "./res_se/v2/best_train_lstm.pt"
    device = torch.device("cuda")
    train_dir = './data/feature/train'
    test_dir = './data/feature/test'
    model = init_model(model_path, device)

    # make speaker classes
    classes = make_classes(train_dir, model, device)
    # classes = load_pickle("classes.pickle")

    utt_list = sorted([f for f in glob.glob(test_dir + '/*.pt') if os.path.isfile(f)])
    spk_id = classification(utt_list, classes, model, device)
    recording = [f.split('/')[-1].replace(".pt", ".wav") for f in utt_list]

    submission = pd.DataFrame({"recording" : recording, "voice_id" : spk_id})
    submission = submission.sort_values(by='recording')
    submission.to_csv("submission.csv", index=False)