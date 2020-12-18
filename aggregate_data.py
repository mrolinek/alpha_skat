import os
import glob
import sys

import numpy as np



def concat_np_files(np_file_list, k):
    file_list = np_file_list if k is None else np_file_list[:k]
    arrs = [np.load(f) for f in file_list]
    return np.concatenate(arrs, axis=0)


def aggregate_all_data(source_dir, target_dir):
    def aggregate(file_pattern, k=None):
        filename = os.path.join(target_dir, file_pattern+'.npy')
        if os.path.isfile(filename):
            print(f"File {filename} exists. Overwrite? (y/N)")
            answer = input()
            if answer.lower() != 'y':
                print("Skipping ...")
                return

        all_files = glob.glob(f"{source_dir}/**/{file_pattern}*.npy", recursive=True)
        big_concat = concat_np_files(sorted(all_files), k)
        np.save(os.path.join(target_dir, file_pattern+'.npy'), big_concat)
        print(file_pattern, " mean: ", big_concat.mean(), " shape: ", big_concat.shape)

    aggregate("inputs")
    aggregate("policy_probs")
    aggregate("masks")
    aggregate("full_states", k=3000)
    aggregate("full_state_values", k=3000)


if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    aggregate_all_data(source_dir, target_dir)
