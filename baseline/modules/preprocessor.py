""" Preprocessor
"""
import random
from multiprocess.pool import ThreadPool
from datetime import datetime

from typing import Optional, Union
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from modules.datasets import Speaker
from modules.params_data import *

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None):
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    return wav


def wav_to_spectrogram(wav):
    frames = np.abs(librosa.core.stft(
        wav,
        n_fft=n_fft,
        hop_length=int(sampling_rate * window_step / 1000),
        win_length=int(sampling_rate * window_length / 1000),
    ))
    return frames.astype(np.float32).T


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


class DatasetLog:
    """
    Registers metadata about the dataset in a text file.
    """

    def __init__(self, root, name):
        self.text_file = open(Path(root, "Log_%s.txt" % name.replace("/", "_")), "w")
        self.sample_data = dict()

        start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Creating dataset %s on %s" % (name, start_time))
        self.write_line("-----")
        self._log_params()

    def _log_params(self):
        from modules import params_data
        self.write_line("Parameter values:")
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            self.write_line("\t%s: %s" % (param_name, value))
        self.write_line("-----")

    def write_line(self, line):
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs):
        for param_name, value in kwargs.items():
            if not param_name in self.sample_data:
                self.sample_data[param_name] = []
            self.sample_data[param_name].append(value)

    def finalize(self):
        self.write_line("Statistics:")
        for param_name, values in self.sample_data.items():
            self.write_line("\t%s:" % param_name)
            self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
            self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
        self.write_line("-----")
        end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
        self.write_line("Finished on %s" % end_time)
        self.text_file.close()


def _init_preprocess_dataset(dataset_name, dataset_root, out_dir) -> (Path, DatasetLog):
    if not dataset_root.exists():
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        return None, None
    return dataset_root, DatasetLog(out_dir, dataset_name)


def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, extension,
                             skip_existing, logger):
    print("%s: Preprocessing train data for %d speakers." % (dataset_name, len(speaker_dirs)))

    # Function to preprocess utterances for one speaker
    def preprocess_speaker(speaker_dir: Path):

        # Create an output directory with that name, as well as a txt file containing a
        # reference to each source file.
        speaker_out_dir = out_dir.joinpath(speaker_dir.name)
        speaker_out_dir.mkdir(exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")

        # There's a possibility that the preprocessing was interrupted earlier, check if
        # there already is a sources file.
        if sources_fpath.exists():
            try:
                with sources_fpath.open("r") as sources_file:
                    existing_fnames = {line.split(",")[0] for line in sources_file}
            except:
                existing_fnames = {}
        else:
            existing_fnames = {}

        # Gather all audio files for that speaker recursively
        sources_file = sources_fpath.open("a" if skip_existing else "w")
        for in_fpath in speaker_dir.glob("*.%s" % extension):
            # Check if the target output file already exists
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            # Load and preprocess the waveform
            wav = preprocess_wav(in_fpath)
            if len(wav) == 0:
                print(in_fpath)
                continue

            # Create the mel spectrogram, discard those that are too short
            # frames = audio.wav_to_mel_spectrogram(wav)
            frames = wav_to_spectrogram(wav)

            # if len(frames) < partials_n_frames:
            #     continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            logger.add_sample(duration=len(wav) / sampling_rate)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))


        sources_file.close()

    # Process the utterances for each speaker

    with ThreadPool(1) as pool:
        list(tqdm(pool.imap(preprocess_speaker, speaker_dirs), dataset_name, len(speaker_dirs),
                  unit="speakers"))
    logger.finalize()
    print("Done preprocessing train dataset : %s.\n" % dataset_name)




def _preprocess_speaker_dirs_test(speaker_files, dataset_name, datasets_root, out_dir, extension,
                             skip_existing, logger):
    print("%s: Preprocessing test data for %d files." % (dataset_name, len(speaker_files)))
    sources_fpath = out_dir.joinpath("_sources.txt")

    # There's a possibility that the preprocessing was interrupted earlier, check if
    # there already is a sources file.
    if sources_fpath.exists():
        try:
            with sources_fpath.open("r") as sources_file:
                existing_fnames = {line.split(",")[0] for line in sources_file}
        except:
            existing_fnames = {}
    else:
        existing_fnames = {}

    # Gather all audio files for that speaker recursively
    sources_file = sources_fpath.open("a" if skip_existing else "w")
    for speaker_file in speaker_files:
        out_fname = speaker_file.name.replace(".%s" % extension, ".npy")

        # Load and preprocess the waveform
        wav = preprocess_wav(speaker_file)
        if len(wav) == 0:
            print(speaker_file)
            continue

        # Create the mel spectrogram, discard those that are too short
        # frames = audio.wav_to_mel_spectrogram(wav)
        frames = wav_to_spectrogram(wav)

        # if len(frames) < partials_n_frames:
        #     continue

        out_fpath = out_dir.joinpath(out_fname)
        np.save(out_fpath, frames)
        logger.add_sample(duration=len(wav) / sampling_rate)
        sources_file.write("%s,%s\n" % (out_fname, speaker_file))

    sources_file.close()
    logger.finalize()
    print("Done preprocessing test dataset : %s.\n" % dataset_name)


def  preprocess_data(dataset_root: Path, partition: str, out_dir: Path, skip_existing=False):
    # Initialize the preprocessing
    dataset_name = "speaker_recognition"

    dataset_root, logger = _init_preprocess_dataset(dataset_name, dataset_root, out_dir)

    if partition !='test':
        speakers = [x for x in dataset_root.joinpath(partition).glob("*") if x.is_dir() and x.name != '.ipynb_checkpoints']
        print("Data : found %d different folders on the disk." %
          (len(set(speakers))))
        _preprocess_speaker_dirs(speakers, dataset_name, dataset_root, out_dir, "wav",
                             skip_existing, logger)
    else:
        wav_files = list(dataset_root.joinpath(partition).glob("*.wav"))
        print("Data : found %d different files on the disk." %
          (len(wav_files)))
        _preprocess_speaker_dirs_test(wav_files, dataset_name, dataset_root, out_dir, "wav",
                                      skip_existing, logger)



def get_train_sources(dataset_dir):
    speaker_dirs = [f for f in dataset_dir.glob("*") if f.is_dir() and f.name != '.ipynb_checkpoints']

    if len(speaker_dirs) == 0:
        raise Exception("No speakers found. Make sure you are pointing to the directory "
                        "containing all preprocessed speaker directories.")
    speakers = list(Speaker(speaker_dir) for speaker_dir in speaker_dirs)

    sources = []
    for speaker in speakers:
        sources.extend(speaker.sources)

    return sources

def get_test_sources(dataset_dir):
    speakers = Speaker(dataset_dir, partition='test', extension='npy')
    sources = speakers.sources
    return sources

def compute_mean_std_all(dataset_dir, output_path_mean, output_path_std):
    print("Computing mean std...")

    train_sources = get_train_sources(dataset_dir)
    test_sources = get_test_sources(dataset_dir)
    print("train files: ", len(train_sources), ", test files: ", len(test_sources))
    assert len(train_sources) != 0, print("train : _sources.txt might be empty ..")
    assert len(test_sources) != 0, print("test :_sources.txt might be empty..")
    sources = train_sources + test_sources
    print("train+test :", len(sources))

    sumx = np.zeros(257, dtype=np.float32)
    sumx2 = np.zeros(257, dtype=np.float32)
    count = 0

    print("computing mean and std for all files")
    for i, source in enumerate(tqdm(sources)):
        try:
            feature = np.load(source[0].joinpath(source[1]))
        except:
            assert False, print(source[0].joinpath(source[1]))

        sumx += feature.sum(axis=0)
        sumx2 += (feature * feature).sum(axis=0)
        count += feature.shape[0]

    mean = sumx / count
    std = np.sqrt(sumx2 / count - mean * mean)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    np.save(output_path_mean, mean)
    np.save(output_path_std, std)