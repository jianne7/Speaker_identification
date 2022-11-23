"""Dataset 정의
"""

from __future__ import print_function


import numpy as np
import torch.utils.data as data
from torchvision import transforms as T
import pandas as pd
from pathlib import Path
import os
from modules.utils import Normalize, TimeReverse, generate_test_sequence


global dataset_root
dataset_root = Path('/data/')

def train_classes(train_dir):
    speakers = [f for f in Path(train_dir).glob("*") if f.is_dir() and f.name !='.ipynb_checkpoints']
    classes = list(set([speaker.name for speaker in speakers]))
    classes.sort()
    classes.append('unknown')
    return classes

def find_classes():
    train_ids = pd.read_csv(dataset_root.joinpath("train_meta.csv"))
    classes = train_ids['id'].tolist()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class Utterance:
    def __init__(self, frames_fpath):
        self.frames_fpath = frames_fpath

    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial utterance of n_frames

        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)


# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path, partition=None, extension='wav'):
        self.root = root
        self.partition = partition
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        self.extension = extension

        if self.partition =='test' or self.partition == None:
            with self.root.joinpath("_sources.txt").open("r") as sources_file:
                sources = [l.strip().split(",") for l in sources_file]
        else:
            with self.root.joinpath("_sources_{}.txt".format(self.partition)).open("r") as sources_file:
                sources = [l.strip().split(",") for l in sources_file]

        # self.root : '..../merged/speaker_id' path (only for train/val), 'merged' path for test
        # frames_fname : something.npy
        # self.name : speaker_id
        # wav_path : wav file path
        self.sources = [[self.root, frames_fname, self.name, wav_path] for frames_fname, wav_path in sources]




    def _load_utterances(self):
        self.utterances = [Utterance(source[0].joinpath(source[1])) for source in self.sources]

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all
        utterances come up at least once every two cycles and in a random order every time.

        :param count: The number of partial utterances to sample from the set of utterances from
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance,
        frames are the frames of the partial utterances and range is the range of the partial
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a


class DeepSpeakerDataset(data.Dataset):

    def __init__(self, data_dir, sub_dir, partial_n_frames, partition=None, is_test=False):
        super(DeepSpeakerDataset, self).__init__()
        self.data_dir = data_dir
        self.root = data_dir.joinpath('feature', sub_dir)
        self.partition = partition
        self.partial_n_frames = partial_n_frames
        self.is_test = is_test

        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir() and f.name != '.ipynb_checkpoints']

        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories. :", self.root)
        self.speakers = [Speaker(speaker_dir, self.partition) for speaker_dir in speaker_dirs]


        classes, class_to_idx = find_classes()
        sources = []
        for speaker in self.speakers:
            sources.extend(speaker.sources)
        self.features = []
        for source in sources:
            item = (source[0].joinpath(source[1]), class_to_idx[source[2]])
            self.features.append(item)
        mean = np.load(self.data_dir.joinpath('mean.npy'))
        std = np.load(self.data_dir.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std),
            TimeReverse(),
        ])

    def load_feature(self, feature_path, speaker_id):
        feature = np.load(feature_path)
        if self.is_test:
            test_sequence = generate_test_sequence(feature, self.partial_n_frames)
            return test_sequence, speaker_id
        else:
            if feature.shape[0] <= self.partial_n_frames:
                start = 0
                while feature.shape[0] < self.partial_n_frames:
                    feature = np.repeat(feature, 2, axis=0)
            else:
                start = np.random.randint(0, feature.shape[0] - self.partial_n_frames)
            end = start + self.partial_n_frames
            return feature[start:end], speaker_id


    def __getitem__(self, index):
        feature_path, speaker_id = self.features[index]
        feature, speaker_id = self.load_feature(feature_path, speaker_id)

        if self.transform is not None:
            feature = self.transform(feature)

        return feature, speaker_id

    def __len__(self):
        return len(self.features)




class DeepSpeakerDataset_test(data.Dataset):

    def __init__(self, data_dir, sub_dir, partial_n_frames, partition='test', is_test=True):
        super(DeepSpeakerDataset_test, self).__init__()
        self.data_dir = data_dir
        self.root = data_dir.joinpath('feature', sub_dir)
        self.partition = partition
        self.partial_n_frames = partial_n_frames
        self.is_test = is_test

        if len(os.listdir(self.root)) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories. :", self.root)

        self.speakers = Speaker(self.root, partition)
        sources = self.speakers.sources
        self.features = []


        for source in sources:
            item = (source[0].joinpath(source[1]))
            self.features.append(item)
        mean = np.load(self.data_dir.joinpath('mean.npy'))
        std = np.load(self.data_dir.joinpath('std.npy'))
        self.transform = T.Compose([
            Normalize(mean, std),
            TimeReverse(),
        ])

    def load_feature(self, feature_path):
        feature = np.load(feature_path)
        if self.is_test:
            test_sequence = generate_test_sequence(feature, self.partial_n_frames)
            return test_sequence
        else:
            if feature.shape[0] <= self.partial_n_frames:
                start = 0
                while feature.shape[0] < self.partial_n_frames:
                    feature = np.repeat(feature, 2, axis=0)
            else:
                start = np.random.randint(0, feature.shape[0] - self.partial_n_frames)
            end = start + self.partial_n_frames
            return feature[start:end]

    def __getitem__(self, index):
        feature_path = self.features[index]
        feature = self.load_feature(feature_path)

        if self.transform is not None:
            feature = self.transform(feature)

        return feature

    def __len__(self):
        return len(self.features)

