"""Predict
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import resnet
from config import cfg, update_config
from modules.recorders import create_logger
from modules.datasets import DeepSpeakerDataset_test
from modules.trainer import predict_identification

global sample_submission_path
sample_submission_path = "/submission/sample_submission.csv"

def parse_args():
    parser = argparse.ArgumentParser(description='Train autospeech network')
    # general
    parser.add_argument('--cfg',
                        default="config/scripts/predict_config.yaml",
                        help='experiment configure file name',
                        # required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--load_path',
                        default="logs/resnet18_iden_2022_06_00_00_00_00/Model/checkpoint_best.pth",
                        help="The path to resumed dir"
                        # , required=True
                        )

    parser.add_argument('--pred_path',
                        default="/submission/prediction.csv",
                        help="The path to prediction.csv"
                        # , required=True
                        )

    parser.add_argument('--pred_threshold',
                        default=0.5,
                        help="The path to resumed dir"
                        # , required=True
                        )

    args = parser.parse_args()

    return args


def main():


    args = parse_args()
    update_config(cfg, args)
    if args.load_path is None:
        raise AttributeError("Please specify load path.")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # SEED
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)


    # model and optimizer

    model = eval('resnet.{}(num_classes={})'.format(cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # resume && make log dir and logger
    if args.load_path and os.path.exists(args.load_path):
        checkpoint = torch.load(args.load_path)

        # load checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        args.path_helper = checkpoint['path_helper']

        logger = create_logger(os.path.dirname(args.load_path))
        logger.info("=> loaded checkpoint '{}'".format(args.load_path))
    else:
        raise AssertionError('Please specify the model to evaluate')
    logger.info(args)
    logger.info(cfg)

    # dataloader
    test_dataset_identification = DeepSpeakerDataset_test(
        Path(cfg.DATASET.DATA_DIR), cfg.DATASET.SUB_DIR, cfg.DATASET.PARTIAL_N_FRAMES, 'test', is_test=True)


    test_loader_identification = torch.utils.data.DataLoader(
        dataset=test_dataset_identification,
        batch_size=1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    #prediction
    recording, prediction = predict_identification(cfg, model, test_loader_identification, criterion, threshold = args.pred_threshold)

    pred_df = pd.DataFrame({"recording": recording, "voice_id": prediction})
    pred_df = pred_df.sort_values(by='recording')
    pred_df.to_csv(args.pred_path, index=False)


    # compare the order of rows with the sample submission
    sample_submission = pd.read_csv(sample_submission_path)
    pred_df = pd.read_csv(args.pred_path)

    if not sample_submission['recording'].equals(pred_df['recording']):
        print("pred doesn't match with the sample submission")


if __name__ == '__main__':
    main()
