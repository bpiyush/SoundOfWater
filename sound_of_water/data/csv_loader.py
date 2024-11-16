"""Utils to load CSV file of audio datasets."""
import os

import pandas as pd
import shared.utils as su


def configure_paths_sound_of_water(
        data_root="/work/piyush/from_nfs2/datasets/SoundOfWater",
    ):
    paths = {
        "data_dir": data_root,
        "video_clip_dir": os.path.join(data_root, "videos"),
        "audio_clip_dir": os.path.join(data_root, "videos"),
        "annot_dir": os.path.join(data_root, "annotations"),
        "split_dir": os.path.join(data_root, "splits"),
    }
    return paths


def load_csv_sound_of_water(
        paths: dict,
        csv_filters=dict(),
        csv_name="localisation.csv",
        ds_name="SoundOfWater",
        split=None,
        check_first_frame_annots=True,
    ):
    """Loads CSV containing metadata of the dataset."""

    su.log.print_update(
        f" [:::] Loading {ds_name}.",
        pos="left",
        fillchar=".",
    )

    # Configure paths
    video_clip_dir = paths["video_clip_dir"]
    audio_clip_dir = paths["audio_clip_dir"]

    # Load main CSV
    path = os.path.join(
        paths["annot_dir"], csv_name,
    )
    assert os.path.exists(path), \
        f"CSV file not found at {path}."
    print(" [:::] CSV path:", path)
    df = pd.read_csv(path)

    # Load side information: containers
    container_path = os.path.join(
        paths['annot_dir'], "containers.yaml",
    )
    assert os.path.exists(container_path)
    containers = su.io.load_yml(container_path)

    # Update CSV with container information (optional)
    update_with_container_info = True
    if update_with_container_info:
        rows = []
        for row in df.iterrows():
            row = row[1].to_dict()
            row.update(containers[row["container_id"]])
            rows.append(row)
        df = pd.DataFrame(rows)
    print(" [:::] Shape of CSV: ", df.shape)

    # 1. Update item_id
    df["item_id"] = df.apply(
        lambda d: f"{d['video_id']}_{d['start_time']:.1f}_{d['end_time']:.1f}",
        axis=1,
    )

    # 2. Update video_clip_path
    # df["video_path"] = df["video_id"].apply(
    #     lambda d: os.path.join(
    #         video_dir, f"{d}.mp4"
    #     )
    # )
    df["video_clip_path"] = df["item_id"].apply(
        lambda d: os.path.join(
            video_clip_dir, f"{d}.mp4"
        )
    )
    df = df[df["video_clip_path"].apply(os.path.exists)]
    print(" [:::] Shape of CSV with available video: ", df.shape)

    # 3. Update audio_clip_path
    # df["audio_path"] = df["video_id"].apply(
    #     lambda d: os.path.join(
    #         audio_dir, f"{d}.mp4"
    #     )
    # )
    df["audio_clip_path"] = df["item_id"].apply(
        lambda d: os.path.join(
            audio_clip_dir, f"{d}.mp4"
        )
    )
    df = df[df["audio_clip_path"].apply(os.path.exists)]
    print(" [:::] Shape of CSV with available audio: ", df.shape)

    # Add first frame annotation paths
    if check_first_frame_annots:
        frame_annot_dir = os.path.join(paths["annot_dir"], "container_bboxes")
        df["box_path"] = df["video_id"].apply(
            lambda d: os.path.join(frame_annot_dir, f"{d}_box.npy"),
        )
        df["mask_path"] = df["video_id"].apply(
            lambda d: os.path.join(frame_annot_dir, f"{d}_mask.npy"),
        )
        df = df[df["box_path"].apply(os.path.exists)]
        df = df[df["mask_path"].apply(os.path.exists)]
        print(" [:::] Shape of CSV with first frame annotations: ", df.shape)

    # Add split filter
    if split is not None and ("item_id" not in csv_filters):
        assert "split_dir" in paths
        split_path = os.path.join(paths["split_dir"], f"{split}")
        assert os.path.exists(split_path), \
            f"Split file not found at {split_path}."
        item_ids = su.io.load_txt(split_path)
        print(" [:::] Number of item_ids in split:", len(item_ids))
        csv_filters["item_id"] = item_ids

    # Apply filter to the CSV
    if len(csv_filters) > 0:
        df = su.pd_utils.apply_filters(df, csv_filters)
        print(" [:::] Shape of CSV after filtering: ", df.shape)
    
    return df


if __name__ == "__main__":
    paths = configure_paths_sound_of_water()
    df = load_csv_sound_of_water(paths)
    row = df.iloc[0].to_dict()
    su.log.json_print(row)
