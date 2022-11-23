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
import glob
import random
import argparse
import numpy as np

from collections import OrderedDict

from tqdm import tqdm
from callbacks import EarlyStopping


N = 64 # the number of speakers per mini-batch
M = 10 # the number of utterences per speaker

    
class AudioBucketDataset(Dataset):
    def __init__(self, root):
        
        self.utts_per_speaker = 10
        self.root_path = root
        
        files = glob.glob('{}/*'.format(self.root_path))
        self.spk_id_list = []
        
        for f in tqdm(files):
            spk_id = f.split('/')[-1]
            self.spk_id_list.append(spk_id)

    def __len__(self):
        return len(self.spk_id_list)
    
    def __getitem__(self, idx: int):
        
        spk_id = self.spk_id_list[idx]
        
        utts = glob.glob('{}/{}/*.pt'.format(self.root_path, spk_id))
        # utt_list = random.sample(utts, self.utts_per_speaker)
        if len(utts) >= 10:
            utt_list = random.sample(utts, 10)
        else:
            utt_list = random.choices(utts, k=10)
        
        mel_specgram_tensor_list = []
        for utt in utt_list:
            mel_specgram_tensor = torch.load(utt) # (1, T, 40)
            mel_specgram_tensor_list.append(mel_specgram_tensor.squeeze(0))
            # print(f'mel_shape:{mel_specgram_tensor.squeeze(0).shape}')
                    
        return mel_specgram_tensor_list


# Dataloader

def audio_collate_fn(batch):

    sampling_frame_length = random.randint(140, 180)
    
    melspec_list = []
    # print(f'sampling_frame_length:{sampling_frame_length}')
    for mel_specgram_tensor_list, _ in batch: # batch_size
        for utt_tensor in mel_specgram_tensor_list: # utterances per speaker
            len_time = utt_tensor.size(0) # time

            if len_time < sampling_frame_length:
                padder = torch.zeros(sampling_frame_length-len_time, utt_tensor.size(1))
                utt_tensor = torch.cat([utt_tensor, padder], dim=0)
                len_time = utt_tensor.size(0)

            len_time = len_time - sampling_frame_length
            # print(f'len_time:{len_time}')
            start_time = random.randint(0, len_time)
            end_time = start_time + sampling_frame_length
            melspec_list.append(utt_tensor[start_time:end_time, :])
    mel_spec_tensors = torch.stack(melspec_list)

    return mel_spec_tensors, 


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


class GE2E_Loss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2E_Loss, self).__init__()
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, device):
        # x shape : (64, 10, 256)

        assert x.size()[1] >= 2

        gsize = x.size()[1] # 10
        centroids = torch.mean(x, 1) # (64, 256)
        stepsize = x.size()[0] # 64

        cos_sim_matrix = []

        for ii in range(0,gsize): 
            idx = [*range(0,gsize)] # 0 ~ 9
            idx.remove(ii)
            exc_centroids = torch.mean(x[:,idx,:], 1) # (64, 256)
            cos_sim_diag    = F.cosine_similarity(x[:,ii,:],exc_centroids) # (64)
            cos_sim         = F.cosine_similarity(x[:,ii,:].unsqueeze(-1), centroids.unsqueeze(-1).transpose(0,2)) # Cos_sim((64, 256, 1), (1, 256, 64)) -> (64, 64)
            cos_sim[range(0,stepsize),range(0,stepsize)] = cos_sim_diag # 대각선만 어사인
            cos_sim_matrix.append(torch.clamp(cos_sim,1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix,dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label = torch.from_numpy(np.asarray(range(0,stepsize))).to(device)
        nloss = self.criterion(cos_sim_matrix.view(-1,stepsize), torch.repeat_interleave(label,repeats=gsize,dim=0).to(device))
        prec1 = accuracy(cos_sim_matrix.view(-1,stepsize).detach(), torch.repeat_interleave(label,repeats=gsize,dim=0).detach(), topk=(1,))[0]
        
        return nloss, prec1


class LSTM(nn.Module):
    def __init__(self,input_size=40, hidden_size=768, num_layers=3, batch_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = nn.Linear(self.hidden_size, 256)
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
 
    def forward(self, x):
#         print(x.shape)
        x = x.squeeze(0)
        self.lstm.flatten_parameters()
        y,_ = self.lstm(x)
        y = self.layers(y[:,-1,:]) # (BS, T, emb_dim)
        
        y = y / torch.norm(y, p=2, dim=1, keepdim=True) # (BS, emb_dim)
        # y = y.sum(1) / y.size(1) # (emb_dim), average pooling over time frames
#         y = torch.mean(y, dim=1)
        
        return y


def get_ge2e_loss():
    GE2E_loss = GE2E_Loss()
    return GE2E_loss


def get_LSTM_model(batch_size, model_path=None):
    input_size = 40
    hidden_size = 768
    num_layers = 3

    LSTM_model = LSTM(input_size, hidden_size, num_layers, batch_size)

    # load the saved model
    if model_path is not None:
        if isinstance(LSTM_model, nn.DataParallel):  # GPU 병렬사용 적용
            LSTM_model.load_state_dict(torch.load(model_path))
        else: # GPU 병렬사용을 안할 경우
            state_dict = torch.load(model_path) 
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`  ## module 키 제거
                new_state_dict[name] = v
            LSTM_model.load_state_dict(new_state_dict)
        # model.load_state_dict(torch.load(model_path), strict=False)
    
    return LSTM_model

def main_worker(rank, args, ngpus_per_node):

    # #setup the distributed backend for managing the distributed training
    torch.distributed.init_process_group(
        backend = 'nccl',
        init_method='tcp://127.0.0.1:2266',
        world_size=ngpus_per_node,
        rank=rank
    )

    device = torch.device('cuda', rank)
    print("device: {}".format(device))
    
    batch_size = N * M
    num_worker = 12
    num_worker = int(num_worker / ngpus_per_node) # num workers per gpu

    # model_path = '/home/ubuntu/speaker_recognition/ujinne/voice_filter/res_se/vL+L0.1_3/99_train_lstm.pt'
    model_path = None
    LSTM_model = get_LSTM_model(batch_size, model_path).to(device)
    GE2E_loss = get_ge2e_loss().to(device)

    SAVE_PATH = f"./res_se/v{args.version}/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    early_stop = EarlyStopping(patience = 30, verbose = True, \
                                model_path = os.path.join(SAVE_PATH, "best_train_lstm.pt"),
                                optimizer_path = os.path.join(SAVE_PATH, 'best_train_lstm_optim.pt') )

    l = []
    for x in LSTM_model.parameters():
        l.append(x)
    for x in GE2E_loss.parameters():
        l.append(x)


    optimizer = optim.Adam(l, lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)

    LSTM_model = nn.parallel.DistributedDataParallel(LSTM_model, device_ids=[rank], output_device=rank)
    
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
    valid_dist_sampler = DistributedSampler(valid_dataset, shuffle=False)
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

    for i in range(epoch):
        
        LSTM_model.train()
        GE2E_loss.train()

        for x in train_dataloader:
            optimizer.zero_grad()
            
            x = x.to(device)

            dvecs = LSTM_model(x)
            dvecs = dvecs.view(N,M,-1)
            L, acc = GE2E_loss(dvecs, device)
            # print(f'acc:{acc}')
            # print(L)
    #         with amp.scale_loss(L, optimizer) as scaled_loss:
    #             scaled_loss.backward()

            L.backward()

            gn = torch.nn.utils.clip_grad_norm_(l, 5.0)

            optimizer.step()
            scheduler.step()

        if rank == 0:
            writer_train.add_scalar('loss/speaker_embedding_train_v{}'.format(args.version), loss.detach().cpu().numpy(), i+offset)
            # writer_train.add_scalar('acc/speaker_embedding_train_v{}_acc'.format(args.version), acc, i+offset)
            print('Epoch[{}]: Train Loss: {}'.format(i, loss.data) , end =' | ')            

            
        ####################################### valid #######################################
        
            
        LSTM_model.eval()
        GE2E_loss.eval()

        with torch.no_grad():
            for x in valid_dataloader:

                x = x.to(device)

                dvecs = LSTM_model(x)
                dvecs = dvecs.view(N,M,-1)
                L, acc = GE2E_loss(dvecs, device)

            if rank == 0:
                writer_valid.add_scalar('loss/speaker_embedding_train_v{}'.format(args.version), loss.detach().cpu().numpy(), i+offset)
                print('Valid Loss: {}'.format(loss.data))
                
                # writer_valid.add_scalar('acc/speaker_embedding_train_v{}_acc'.format(args.version), acc, i+offset)
                # print('Valid acc:', acc)

            if rank == 0 and i % 100 == 99:
                SAVE_PATH = f"./res_se/v{args.version}/"
                os.makedirs(SAVE_PATH, exist_ok=True)
                
                torch.save(LSTM_model.state_dict(), os.path.join(SAVE_PATH, f"{i+offset}_train_lstm.pt"))
                # print(GE2E_loss.w, GE2E_loss.b)
                print('lr: {: .8f}'.format(optimizer.param_groups[0]['lr']))
                torch.save(optimizer.state_dict(), os.path.join(SAVE_PATH, f'{i+offset}_train_lstm_optim.pt'))
            
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


