import h5py
import numpy as np

np.set_printoptions(precision=9, suppress=True)

def inspect_hdf5_file(hdf5_path: str) -> None:
    '''
        查看 HDF5 中的字段和数据类型
    '''
    def print_item(name: str, obj):
        display_name = '/' + name if not name.startswith('/') else name

        if isinstance(obj, h5py.Dataset):
            ds: h5py.Dataset = obj
            print(f"   Dataset: {display_name}")
            print(f"    • shape = {ds.shape}    • dtype = {ds.dtype}")

            vlen_base = h5py.check_dtype(vlen=ds.dtype)
            if vlen_base is not None:
                n = ds.shape[0]
                lengths = [len(ds[i]) for i in range(n)]
                print(f"    • vlen base dtype = {vlen_base}")
                print(f"    • count = {n}, lengths (min/mean/max) = "
                      f"{min(lengths)}/{int(np.mean(lengths))}/{max(lengths)}")

    print(f"\n Inspecting HDF5 file: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(print_item)


def show_hdf5_data(hdf5_path: str, topic_name: str, lines: int = 5) -> None:
    with h5py.File(hdf5_path, 'r') as f:
        dset = f[topic_name]
        print("\nshape =", dset.shape)
        print("dtype =", dset.dtype)
        data = dset[...]
        print(f"前 {lines} 行：")
        for i in range (lines):
            print(data[i])


if __name__ == '__main__':
    hdf5_dir = 'data/hdf5_0613'
    for i in range(30):
        hdf5_path = f'{hdf5_dir}/episode_{i}.hdf5'
        # inspect_hdf5_file(hdf5_path)
        show_hdf5_data(hdf5_path, '/observation/state', lines=10)
        show_hdf5_data(hdf5_path, '/action/actions', lines=10)
        exit()
