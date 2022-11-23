"""utils
"""

from tqdm import tqdm
import pandas as pd

import torch
import dateutil.tz

import os

from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import numpy
import random


plt.switch_backend('agg')

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def find_id(dataset_root, idx):
    train_ids = pd.read_csv(dataset_root.joinpath("train_meta.csv"))
    classes = train_ids['id'].tolist()
    return classes[idx]

# Split train dataset into train & validation
def split_train_val(train_dir, split_path, val_ratio=0.2):
    speakers = [x for x in train_dir.iterdir() if x.is_dir()]
    print("Found ", len(speakers), "speakers")
    wav_list = []
    split_list = []
    for id_ in tqdm(speakers):
        utterences = ['/'.join(x.parts[-2:]) for x in id_.glob("*.wav")]
        split_list.extend((np.random.rand(len(utterences)) > val_ratio).astype(int))
        wav_list.extend(utterences)
    # save split.txt
    with open(split_path, "w") as split:
        for i, n in enumerate(split_list):
            split.write("%d %s\n" % (n, wav_list[i]))

    print(len(split_list)," files split into train/val")

def create_train_meta(train_dir, train_meta_path):
    speakers = [x.name for x in train_dir.iterdir() if x.is_dir() and x.name!='.ipynb_checkpoints']
    assert len(speakers)!=0, print(train_dir, " extraction failed")
    np.sort(speakers)
    speakers.append('unknown')
    train_meta = pd.DataFrame({"id":speakers})
    train_meta.to_csv(train_meta_path, index=False)
    print(train_meta_path, " saved.")


def partition_voxceleb(feature_root, split_txt_path):
    print("partitioning VoxCeleb...")
    with open(split_txt_path, 'r') as f:
        split_txt = f.readlines()
    train_set = []
    val_set = []
    for line in split_txt:
        items = line.strip().split()
        if items[0] == '1':
            train_set.append(items[1])
        elif items[0] == '0':
            val_set.append(items[1])
        else:
            print(split_txt_path, " might contain invalid split indicator (ex. 1: train, 0: val)")

    assert len(train_set) != 0
    assert len(val_set) != 0

    speakers = os.listdir(feature_root)
    for speaker in tqdm(speakers):
        speaker_dir = os.path.join(feature_root, speaker)
        if not os.path.isdir(speaker_dir) or speaker=='.ipynb_checkpoint':
            continue

        with open(os.path.join(speaker_dir, '_sources.txt'), 'r') as f:
            speaker_files = f.readlines()

        train = []
        val = []

        for line in speaker_files:
            address = line.strip().split(',')[1]
            fname = os.path.join(*address.split('/')[-2:])
            if fname in val_set:
                val.append(line)
            elif fname in train_set:
                train.append(line)
            else:
                print("fname neither in train nor val")

        with open(os.path.join(speaker_dir, '_sources_train.txt'), 'w') as f:
            f.writelines('%s' % line for line in train)
        with open(os.path.join(speaker_dir, '_sources_val.txt'), 'w') as f:
            f.writelines('%s' % line for line in val)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_pretrained_weights(model, checkpoint):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    checkpoint_file = torch.load(checkpoint)
    pretrain_dict = checkpoint_file['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger:
            self.logger.info('\t'.join(entries))
        else:
            print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def set_path(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    """
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if torch.isnan(ret).sum():
        import ipdb
        ipdb.set_trace()
        raise OverflowError(f'gumbel softmax output: {ret}')
    return ret

class Normalize(object):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, input):
        return (input - self.mean) / self.std



class TimeReverse(object):
    def __init__(self, p=0.5):
        super(TimeReverse, self).__init__()
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return np.flip(input, axis=0).copy()
        return input


def generate_test_sequence(feature, partial_n_frames, shift=None):
    while feature.shape[0] <= partial_n_frames:
        feature = np.repeat(feature, 2, axis=0)
    if shift is None:
        shift = partial_n_frames // 2
    test_sequence = []
    start = 0
    while start + partial_n_frames <= feature.shape[0]:
        test_sequence.append(feature[start: start + partial_n_frames])
        start += shift
    test_sequence = np.stack(test_sequence, axis=0)
    return test_sequence