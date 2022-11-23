"""preprocess
"""


import argparse
import subprocess
from modules.utils import *
from modules.preprocessor import *


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description="Preprocesses audio files from datasets.",
        formatter_class=MyFormatter
    )
    parser.add_argument("--dataset_root", type=Path,
                        default="/data/",
                        help= \
        "Path to the directory containing voice_recognition datasets. It should be arranged as:")

    parser.add_argument("-s",
                        "--skip_existing",
                        action="store_true",
                        help= \
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")

    args = parser.parse_args()

    # Process the arguments
    train_out_dir = args.dataset_root.joinpath("feature", "train")
    test_out_dir = args.dataset_root.joinpath("feature", "test")
    merged_out_dir = args.dataset_root.joinpath("feature", "merged")
    assert args.dataset_root.exists()
    train_out_dir.mkdir(exist_ok=True, parents=True)
    test_out_dir.mkdir(exist_ok=True, parents=True)
    merged_out_dir.mkdir(exist_ok=True, parents=True)

    # Create split.txt and train_meta.csv
    split_path = args.dataset_root.joinpath('split.txt')
    train_meta_path = args.dataset_root.joinpath('train_meta.csv')
    split_train_val(args.dataset_root.joinpath("train"), split_path, val_ratio = 0.2)
    create_train_meta(args.dataset_root.joinpath("train"),train_meta_path)
    assert split_path.exists()
    assert train_meta_path.exists()

    #Preprocess the datasets
    preprocess_data(args.dataset_root, 'train', train_out_dir, args.skip_existing)
    preprocess_data(args.dataset_root, 'test', test_out_dir, args.skip_existing)
    print("...Copying ", train_out_dir, " to ", merged_out_dir)
    for path in train_out_dir.iterdir():
        subprocess.call(['cp', '-r', path.as_posix(), merged_out_dir.as_posix()])
    print("...Copying ", test_out_dir, " to ", merged_out_dir)
    for path in test_out_dir.iterdir():
        subprocess.call(['cp', '-r', path.as_posix(), merged_out_dir.as_posix()])
    compute_mean_std_all(merged_out_dir, args.dataset_root.joinpath('mean.npy'),
                     args.dataset_root.joinpath('std.npy'))
    partition_voxceleb(merged_out_dir, split_path)
    print("Preprocess finished")







