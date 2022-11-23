import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torch.backends.cudnn as cudnn

# from apex import amp
# from apex.parallel import DistributedDataParallel

import torchaudio
import torchaudio.transforms as T

import os
import gc
import glob
import random
import argparse
import numpy as np

from collections import OrderedDict

import json
from tqdm import tqdm
from callbacks import EarlyStopping

from model.transformer_model import SpeakerTransformer, LabelSmoothingLoss

gc.collect()
torch.cuda.empty_cache()

N = 100


def index2char(speaker_id: list):
    txt2id = {f'{txt}':i for i, txt in enumerate(speaker_id)}
    # txt2id['unknown'] = len(txt2id)+1
    id2txt = {int(value):key for key, value in txt2id.items()}

    return txt2id, id2txt


class AudioBucketDataset(Dataset):
    def __init__(self, root):
        
        self.utts_per_speaker = 10
        self.root_path = root
        self.normalize = True

        files = glob.glob('{}/*'.format(self.root_path))
        self.spk_id_list = []
        
        for f in tqdm(files):
            spk_id = f.split('/')[-1]
            self.spk_id_list.append(spk_id)

        self.txt2id, _ = index2char(self.spk_id_list)
        
    def __len__(self):
        return len(self.spk_id_list)
    
    def __getitem__(self, idx: int):
        spk_id = self.spk_id_list[idx]
        
        utts = glob.glob('{}/{}/*.pt'.format(self.root_path, spk_id))
        # utt_list = random.sample(utts, self.utts_per_speaker)
        utt = random.choice(utts)
        
        mel_specgram_tensor = torch.load(utt) # (1, T, 40)
        mel_specgram_tensor = mel_specgram_tensor.squeeze(0)
        # print(f'mel_shape:{mel_specgram_tensor.squeeze(0).shape}')
        if self.normalize:
            mean = torch.mean(mel_specgram_tensor, 0, keepdim=True)
            std = torch.std(mel_specgram_tensor, 0, keepdim=True)

            mel_specgram_tensor = (mel_specgram_tensor - mean) / (std + 1e-8)


        label = int(self.txt2id.get(spk_id))
        
        return mel_specgram_tensor, label


# Dataloader

def audio_collate_fn(batch):
    sampling_frame_length = random.randint(140, 180)
    
    melspec_list = []
    labels = []
    # print(f'sampling_frame_length:{sampling_frame_length}')
    for mel_specgram_tensor, label in batch: # batch_size
        
        len_time = mel_specgram_tensor.size(0) # time

        if len_time < sampling_frame_length:
            padder = torch.zeros(sampling_frame_length-len_time, mel_specgram_tensor.size(1))
            mel_specgram_tensor = torch.cat([mel_specgram_tensor, padder], dim=0)
            len_time = mel_specgram_tensor.size(0)

        len_time = len_time - sampling_frame_length
        # print(f'len_time:{len_time}')
        start_time = random.randint(0, len_time)
        end_time = start_time + sampling_frame_length
            
        melspec_list.append(mel_specgram_tensor[start_time:end_time, :])
        labels.append(label)

    mel_spec= torch.stack(melspec_list)
    mel_spec_len = torch.LongTensor([mel_spec.size(0) for mel_spec in melspec_list])
    labels = torch.LongTensor(labels)

    return mel_spec,  mel_spec_len, labels


def  accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() #transpose
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def main_worker(rank, args, ngpus_per_node):

    #setup the distributed backend for managing the distributed training
    torch.distributed.init_process_group(
                                            backend = 'nccl',
                                            init_method='tcp://127.0.0.1:2266',
                                            world_size=ngpus_per_node,
                                            rank=rank
                                        )

    device = torch.device('cuda', rank)
    print("device: {}".format(device))
    

    spk_ids = sorted([f.split('/')[-1] for f in glob.glob(args.data_path + '/*') if os.path.isdir(f)])
    txt2id, _ = index2char(spk_ids)
    
    batch_size = N
    num_worker = 12
    num_worker = int(num_worker / ngpus_per_node) # num workers per gpu

    output_size = len(txt2id)
    model = SpeakerTransformer(device, output_size).to(device)

    SAVE_PATH = f"./res_se/v{args.version}/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    early_stop = EarlyStopping(patience = 30, verbose = True, \
                                model_path = os.path.join(SAVE_PATH, "best_train_transformer.pt"),
                                optimizer_path = os.path.join(SAVE_PATH, 'best_train_transformer_optim.pt') )

    l = []
    for x in model.parameters():
        l.append(x)

    optimizer = optim.Adam(l, lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingLoss(output_size, smoothing=0.1, dim=-1)
    # criterion = LabelSmoothingLoss(size=len(txt2id), padding_idx=0, smoothing=0.1)


    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    train_dataset = AudioBucketDataset(args.data_path)
    train_dist_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=N,
                                  collate_fn=audio_collate_fn,
                                  pin_memory=True,
                                  num_workers=num_worker,
                                  sampler=train_dist_sampler,
                                  drop_last=True
                                  )
    
    valid_dataset = AudioBucketDataset(args.valid_path)
    valid_dist_sampler = DistributedSampler(valid_dataset, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=N,
                                  collate_fn=audio_collate_fn,
                                  pin_memory=True,
                                  num_workers=num_worker,
                                  sampler=valid_dist_sampler,
                                  drop_last=True
                                  )

    writer_train = SummaryWriter('./runs_se/speaker_embedding_v{}_train'.format(args.version))
    writer_valid = SummaryWriter('./runs_se/speaker_embedding_v{}_valid'.format(args.version))

    epoch = 200000
    offset = 0

    for i in tqdm(range(epoch)):
        
        # LSTM_model.train()
        model.train()
        for mel, mel_len, labels in train_dataloader:
            # print(f'x:{x}')
            optimizer.zero_grad()
            
            mel = mel.to(device)
            mel_len = mel_len.to(device)
            target = labels.to(device)

            output = model(mel, mel_len)
            # print(f'output:{output.shape}')

            loss = criterion(output.contiguous(), target.contiguous())
            # print(f'loss:{loss}')
            loss.backward()
            # L.backward()

            gn = torch.nn.utils.clip_grad_norm_(l, 5.0)

            optimizer.step()
            scheduler.step()

            del mel
            del target

            torch.cuda.empty_cache()

        if rank == 0:
            writer_train.add_scalar('loss/speaker_embedding_train_v{}'.format(args.version), loss.detach().cpu().numpy(), i+offset)
            # writer_train.add_scalar('acc/speaker_embedding_train_v{}_acc'.format(args.version), acc, i+offset)
            print('Epoch[{}]: Train Loss: {}'.format(i, loss.data) , end =' | ')            

            
        ####################################### valid #######################################
        
            
        model.eval()

        with torch.no_grad():
            valid_losses = []
            for mel, mel_len, labels in valid_dataloader:

                mel = mel.to(device)
                target = labels.to(device)

                output = model(mel, mel_len)
                # output = output.view(N,M,-1)
                loss = criterion(output.contiguous(), target.contiguous())

                # valid_losses.append(L)
                valid_losses.append(loss)
                
                del mel
                del target

                torch.cuda.empty_cache()

            if rank == 0:
                writer_valid.add_scalar('loss/speaker_embedding_train_v{}'.format(args.version), loss.detach().cpu().numpy(), i+offset)
                print('Valid Loss: {}'.format(loss.data))
                
                # writer_valid.add_scalar('acc/speaker_embedding_train_v{}_acc'.format(args.version), acc, i+offset)
                # print('Valid acc:', acc)

            if rank == 0 and i % 100 == 99:
                SAVE_PATH = f"./res_se/v{args.version}/"
                os.makedirs(SAVE_PATH, exist_ok=True)
                
                torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"{i+offset}_train_transformer.pt"))
                # print(GE2E_loss.w, GE2E_loss.b)
                print('lr: {: .8f}'.format(optimizer.param_groups[0]['lr']))
                torch.save(optimizer.state_dict(), os.path.join(SAVE_PATH, f'{i+offset}_train_transformer_optim.pt'))
            
            # valid_loss = np.average(torch.stack(valid_losses).detach().cpu().numpy()[0])
            
            # early_stop(valid_loss, LSTM_model, optimizer)

            # if early_stop.early_stop:
            #     print("Early stopping")
            #     break

    
            
def main():
    seed = 142
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node

    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--data_path", type= str, required=True, default='/data/feature/train/', help="Preprocessed train data root path")
    parser.add_argument("--valid_path", type= str, required=True, default='/data/feature/valid/', help="Preprocessed valid data root path")
    parser.add_argument("--version", type= str, required=True, default='', help="experiment version")
    args = parser.parse_args()

    world_size = ngpus_per_node
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(args, ngpus_per_node, ))
            
if __name__ == '__main__':
    main()
