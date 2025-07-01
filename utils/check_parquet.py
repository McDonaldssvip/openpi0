import os
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDatasetMetadata

import numpy as np
import matplotlib.pyplot as plt


def check_data(repo_id, num_samples=5, out_dir="./output"):
    root = HF_LEROBOT_HOME / repo_id
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    print(f"\n== Repository: {repo_id} ==")
    print(meta)

    episode_index = 0
    parquet_rel = meta.get_data_file_path(episode_index)
    parquet_path = root / parquet_rel
    print("Parquet 文件路径:", parquet_path)

    hf_ds = load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    )
    print("字段名:", hf_ds.column_names)
    print(f"Episode {episode_index} 总条数: {len(hf_ds)}")

    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(num_samples, len(hf_ds))):
        sample = hf_ds[i]
        print(f"\n--- Sample {i} ---")
        print(sample)
        for key in sample:
            print(f"  {key}: type={type(sample[key])}")
        if 'observation' in sample:
            print("  observation 子字段:", list(sample['observation'].keys()))


        # 若要可视化 image，可以取消下面注释
        # img = sample.get('observation', {}).get('image') or sample.get('image')
        # if img is not None:
        #     arr = np.array(img)
        #     plt.figure(figsize=(4, 3))
        #     plt.imshow(arr)
        #     plt.axis('off')
        #     fn = os.path.join(out_dir, f"{repo_id.replace('/', '_')}_sample_{i}.png")
        #     plt.savefig(fn, bbox_inches="tight", pad_inches=0)
        #     plt.close()
        #     print(f"  已保存图像到: {fn}")




if __name__ == "__main__":
    repos = ['zzh/ranger_0613']
    for repo in repos:
        check_data(repo, num_samples=10)
    # repos = ['zzh/libero']
    # for repo in repos:
    #     check_data(repo, num_samples=5)
