def download_checkpoint():
    from openpi.shared import download
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_base")

def download_libera_data():
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="openvla/modified_libero_rlds", repo_type="dataset",
                    local_dir="data/",allow_patterns="libero_10_no_noops/**")

if __name__ == "__main__":
    '''官方示例使用的数据集，先跑示例了解数据格式'''
    download_libera_data()
    download_checkpoint()