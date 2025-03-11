import sys

sys.path.append("../")
from constants import INCLUDED_CHANNELS, INCLUDED_CHANNELS_BIDS, FREQUENCY, FREQUENCY_BIDS
from data_utils import resampleData, getEDFsignals, getOrderedChannels, getSETsignals, get_BIDS_OrderedChannels
from tqdm import tqdm
import argparse
import numpy as np
import os
import pyedflib
import h5py
import scipy
import mne


def resample_all(raw_data_dir, save_dir):
    print(raw_data_dir)
    print(save_dir)
    data_files = []
    for path, subdirs, files in os.walk(raw_data_dir):
        for name in files:
            if ".edf" in name or ".set" in name:
                data_files.append(os.path.join(path, name))

    failed_files = []
    for idx in tqdm(range(len(data_files))):
        data_fn = data_files[idx]
        save_fn = os.path.join(save_dir, data_fn.split("\\")[-1].split(".")[0] + ".h5")
        if os.path.exists(save_fn):
            continue
        try:
            if ".edf" in data_fn:
                f = pyedflib.EdfReader(data_fn)
                orderedChannels = getOrderedChannels(
                    data_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
                )
                signals = getEDFsignals(f)
                signal_array = np.array(signals[orderedChannels, :])
                sample_freq = f.getSampleFrequency(0)

            elif ".set" in data_fn:
                raw = mne.io.read_raw_eeglab(data_fn, preload=True)
                orderedChannels = get_BIDS_OrderedChannels(
                    data_fn, False, raw.ch_names, INCLUDED_CHANNELS_BIDS
                )
                signals, sample_freq = getSETsignals(data_fn)
                signal_array = np.array(signals[orderedChannels, :])

            else:
                continue

            if sample_freq != FREQUENCY_BIDS:
                signal_array = resampleData(
                    signal_array,
                    to_freq=FREQUENCY_BIDS,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
                hf.create_dataset("resample_freq", data=FREQUENCY_BIDS)

        except BaseException:
            failed_files.append(data_fn)
    if failed_files:
        print("以下文件处理失败:")
        for failed_file in failed_files:
            print(failed_file)
    else:
        print("所有文件处理成功.")

    print("DONE. {} files failed.".format(len(failed_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resample.")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="Full path to raw data files (edf or set).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Full path to dir to save resampled signals.",
    )
    args = parser.parse_args()

    resample_all(args.raw_data_dir, args.save_dir)
