"""Uploads dataset to huggingface datasets."""
import os
import sys

import pandas as pd
import numpy as np

from huggingface_hub import HfApi
import shared.utils as su
from sound_of_water.data.csv_loader import (
    load_csv_sound_of_water,
    configure_paths_sound_of_water,
)


if __name__ == "__main__":
    api = HfApi()

    data_root = "/work/piyush/from_nfs2/datasets/SoundOfWater"
    repo_id = "bpiyush/sound-of-water"

    save_splits = False
    if save_splits:
        # Load CSV
        paths = configure_paths_sound_of_water(data_root)
        df = load_csv_sound_of_water(paths)
        del df["video_clip_path"]
        del df["audio_clip_path"]
        del df["box_path"]
        del df["mask_path"]

        # Splits
        train_ids = su.io.load_txt(os.path.join(data_root, "splits/train.txt"))
        df_train = df[df.item_id.isin(train_ids)]
        df_train["file_name"] = df_train["item_id"].apply(lambda x: f"videos/{x}.mp4")
        df_train.to_csv(os.path.join(data_root, "splits/train.csv"), index=False)
        print(" [:::] Train split saved.")

        test_I_ids = su.io.load_txt(os.path.join(data_root, "splits/test_I.txt"))
        df_test_I = df[df.item_id.isin(test_I_ids)]
        df_test_I["file_name"] = df_test_I["item_id"].apply(lambda x: f"videos/{x}.mp4")
        df_test_I.to_csv(os.path.join(data_root, "splits/test_I.csv"), index=False)
        print(" [:::] Test I split saved.")

        test_II_ids = su.io.load_txt(os.path.join(data_root, "splits/test_II.txt"))
        df_test_II = df[df.item_id.isin(test_II_ids)]
        df_test_II["file_name"] = df_test_II["item_id"].apply(lambda x: f"videos/{x}.mp4")
        df_test_II.to_csv(os.path.join(data_root, "splits/test_II.csv"), index=False)
        print(" [:::] Test II split saved.")

        test_III_ids = su.io.load_txt(os.path.join(data_root, "splits/test_III.txt"))
        df_test_III = df[df.item_id.isin(test_III_ids)]
        df_test_III["file_name"] = df_test_III["item_id"].apply(lambda x: f"videos/{x}.mp4")
        df_test_III.to_csv(os.path.join(data_root, "splits/test_III.csv"), index=False)
        print(" [:::] Test III split saved.")


    create_splits = False
    if create_splits:
        train_ids = su.io.load_txt(os.path.join(data_root, "splits/train.txt"))
        train_ids = np.unique(train_ids)

        test_I_ids = su.io.load_txt(os.path.join(data_root, "splits/test_I.txt"))
        test_I_ids = np.unique(test_I_ids)

        other_ids = np.array(
            list(set(df.item_id.unique()) - set(train_ids) - set(test_I_ids))
        )
        sub_df = df[~df.item_id.isin(set(train_ids) | set(test_I_ids))]
        X = sub_df[
            (sub_df.visibility != "transparent") & (sub_df["shape"].isin(["cylindrical", "semiconical"]))
        ]
        test_II_ids = list(X.item_id.unique())
        assert set(test_II_ids).intersection(set(train_ids)) == set()
        assert set(test_II_ids).intersection(set(test_I_ids)) == set()
        su.io.save_txt(test_II_ids, os.path.join(data_root, "splits/test_II.txt"))

        X = sub_df[ 
            (sub_df.visibility.isin(["transparent", "opaque"])) & \
            (sub_df["shape"].isin(["cylindrical", "semiconical", "bottleneck"]))
        ]
        test_III_ids = list(X.item_id.unique())
        assert set(test_III_ids).intersection(set(train_ids)) == set()
        assert set(test_III_ids).intersection(set(test_I_ids)) == set()
        assert set(test_III_ids).intersection(set(test_II_ids)) != set()
        su.io.save_txt(test_III_ids, os.path.join(data_root, "splits/test_III.txt"))

    upload_file = True
    if upload_file:
        file = "README.md"
        print(f" [:::] Uploading file: {file}")
        api.upload_file(
            path_or_fileobj=os.path.join(data_root, file),
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="dataset",
        )

    upload_folder = False
    if upload_folder:
        # Upload splits folder
        foldername = "annotations"
        print(f" [:::] Uploading folder: {foldername}")
        api.upload_folder(
            folder_path=os.path.join(data_root, foldername),
            path_in_repo=foldername, # Upload to a specific folder
            repo_id=repo_id,
            repo_type="dataset",
        )
