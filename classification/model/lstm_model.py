import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # self.layers = nn.Linear(self.hidden_size, 256)
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.layers = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
#         print(x.shape)
        x = x.squeeze(0)
        self.lstm.flatten_parameters()
        y,_ = self.lstm(x)
        y = self.layers(y[:,-1,:]) # (BS, T, emb_dim)
        
        # y = y / torch.norm(y, p=2, dim=1, keepdim=True) # (BS, emb_dim)
        # y = y.sum(1) / y.size(1) # (emb_dim), average pooling over time frames
#         y = torch.mean(y, dim=1)
        
        return y

    def predict(self, x):
        # print(f'x_shape : {x.shape}')
        x = x.transpose(0, 1) # ex) (333, 40) -> (40, 333)
        # print(f'x_shape : {x.shape}')
        x = x.unfold(1, 160, 80) # (num_mels, T', window)  ex) 40, 3, 160
        # print(f'x_shape : {x.shape}')
        x = x.permute(1, 2, 0) # (T', window, num_mels))   ex) 3, 160, 40
        
        y,_ = self.lstm(x)
        y = y[:, -1, :] # (T', lstm_hidden), use last frame only
        y = self.layers(y) # (BS, T, emb_dim)
        y = y / torch.norm(y, p=2, dim=1, keepdim=True) # (BS, T, emb_dim)
        y = torch.mean(y, dim=0)
        y = y.unsqueeze(0)
        
        return y


class LabelSmoothingLoss(nn.Module):
    def __init__(self, output_size, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = output_size
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))