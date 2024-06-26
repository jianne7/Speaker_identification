"""학습 스크립트
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import resnet
from config import cfg, update_config
from modules.utils import set_path, count_parameters
from modules.datasets import DeepSpeakerDataset
from modules.trainer import train_from_scratch, validate_identification
from modules.losses import CrossEntropyLoss
from modules.recorders import save_checkpoint, create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train energy network')
    # general
    parser.add_argument('--cfg',
                        default="config/scripts/resnet18_iden.yaml",
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--load_path',
                        help="The path to resumed dir",
                        default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Set the random seed manually for reproducibility.
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    # model and optimizer
    model = eval('resnet.{}(num_classes={})'.format(
        cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES))
    model = model.cuda()
    optimizer = optim.Adam(
        model.net_parameters() if hasattr(model, 'net_parameters') else model.parameters(),
        lr=cfg.TRAIN.LR
    )

    # Loss
    criterion = CrossEntropyLoss(cfg.MODEL.NUM_CLASSES).cuda()

    # resume && make log dir and logger
    if args.load_path and os.path.exists(args.load_path):
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint_best.pth')
        # checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        # load checkpoint
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.path_helper = checkpoint['path_helper']

        logger = create_logger(args.path_helper['log_path'])
        logger.info("=> loaded checkpoint '{}'".format(checkpoint_file))
    else:
        exp_name = args.cfg.split('/')[-1].split('.')[0]
        args.path_helper = set_path('logs', exp_name)
        logger = create_logger(args.path_helper['log_path'])
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        best_acc1 = 0.0
        last_epoch = -1
    logger.info(args)
    logger.info(cfg)
    logger.info("Number of parameters: {}".format(count_parameters(model)))

    # dataloader
    train_dataset = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, 'train')

    val_dataset_identification = DeepSpeakerDataset(
        Path(cfg.DATASET.DATA_DIR),  cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, 'val', is_test=True)


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    val_loader_identification = torch.utils.data.DataLoader(
        dataset=val_dataset_identification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )


    # training setting
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // cfg.VAL_FREQ,
    }

    # training loop
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, cfg.TRAIN.LR_MIN,
        last_epoch=last_epoch
    )

    for epoch in tqdm(range(begin_epoch, cfg.TRAIN.END_EPOCH), desc='train progress'):
        model.train()
        train_from_scratch(cfg, model, optimizer, train_loader, criterion, epoch, writer_dict, lr_scheduler)
        if epoch % cfg.VAL_FREQ == 0:
            acc = validate_identification(cfg, model, val_loader_identification, criterion)

            # remember best acc@1 and save checkpoint
            is_best = acc > best_acc1
            best_acc1 = max(acc, best_acc1)

            # save
            logger.info('=> saving checkpoint to {}'.format(args.path_helper['ckpt_path']))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'path_helper': args.path_helper
            }, is_best, args.path_helper['ckpt_path'], 'checkpoint_{}.pth'.format(epoch))
        lr_scheduler.step(epoch)


if __name__ == '__main__':
    main()
