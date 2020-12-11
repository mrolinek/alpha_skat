import os
import glob
import sys

import numpy as np



def concat_np_files(np_file_list):
    arrs = [np.load(f) for f in np_file_list]
    return np.concatenate(arrs, axis=0)


def aggregate_all_data(source_dir, target_dir):
    def aggregate(file_pattern):
        all_files = glob.glob(f"{source_dir}/**/{file_pattern}*.npy", recursive=True)
        big_concat = concat_np_files(sorted(all_files))
        np.save(os.path.join(target_dir, file_pattern), big_concat)
        print(file_pattern, " mean: ", big_concat.mean(), " shape: ", big_concat.shape)

    aggregate("inputs")
    aggregate("policy_probs")
    aggregate("masks")
    aggregate("state_values")


if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    aggregate_all_data(source_dir, target_dir)
