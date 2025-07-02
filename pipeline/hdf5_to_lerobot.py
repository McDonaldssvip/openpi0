import h5py
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

def convert(data_dir: Path, repo_id: str, fps: int = 10):

    features = {
        "image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (17,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (17,),
            "names": ["actions"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="realman",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for h5file in sorted(Path(data_dir).glob("episode_*.hdf5")):
        with h5py.File(h5file, "r") as f:
            obs_grp = f.get("observation") or f
            act_grp = f.get("action") or f
            num_steps = obs_grp["image"].shape[0]

            for i in range(num_steps):
                original_image = Image.fromarray(obs_grp["image"][i])
                image_resized = original_image.resize((256, 256), Image.Resampling.LANCZOS)
                image_resized_array = np.array(image_resized)
                wrist_image_array = np.zeros((256, 256, 3), dtype=np.uint8)
                task_name = 'move to the table and pick up the object'

                frame = {
                    "image": image_resized_array,
                    "state": obs_grp["state"][i],
                    "actions": act_grp["actions"][i],
                    "wrist_image": wrist_image_array,
                    "task": task_name
                }
                dataset.add_frame(frame)
            dataset.save_episode()

    print(f"✅ Converted LeRobot dataset saved to directory '{HF_LEROBOT_HOME / repo_id}'")


if __name__ == "__main__":
    DATA_DIR    = Path("./data/hdf5_0613")
    REPO_ID     = "zzh/ranger_0613"
    FPS         = 10

    convert(DATA_DIR, REPO_ID, FPS)
