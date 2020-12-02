import os
import glob
import sys

import numpy as np



def concat_np_files(np_file_list):
    arrs = [np.load(f) for f in np_file_list]
    return np.concatenate(arrs, axis=0)


def aggregate_all_data(source_dir, target_dir):
    all_state_files = glob.glob(f"{source_dir}/**/inputs*.npy", recursive=True)
    all_states = concat_np_files(sorted(all_state_files))

    print(all_states.mean())

    all_mask_files = glob.glob(f"{source_dir}/**/masks*.npy", recursive=True)
    all_masks = concat_np_files(sorted(all_mask_files))

    all_probs_files = glob.glob(f"{source_dir}/**/probs*.npy", recursive=True)
    all_probs = concat_np_files(sorted(all_probs_files))

    print(all_states.shape)
    print(all_masks.shape)
    print(all_probs.shape)
    np.save(os.path.join(target_dir, "inputs.npy"), all_states)
    np.save(os.path.join(target_dir, "masks.npy"), all_masks)
    np.save(os.path.join(target_dir, "probs.npy"), all_probs)

if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    aggregate_all_data(source_dir, target_dir)
