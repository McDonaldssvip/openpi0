import h5py
import numpy as np

'''
    1.将数据中的state从角度转换为弧度
    2.增加action字段
'''

def convert_angle_to_radian(hdf5_path: str) -> None:
    """
    将 HDF5 文件中 /observation/state 的角度字段（1-6、8-13 维）从角度转换为弧度，并直接写回文件
    第 7 维（idx=6）和第 14 维（idx=13）保持不变
    """
    angle_indices = list(range(0, 6)) + list(range(7, 13))
    
    with h5py.File(hdf5_path, 'r+') as f:
        ds_state = f['/observation/state']
        data_state = ds_state[...].astype(np.float64)
        data_state[..., angle_indices] = np.deg2rad(data_state[..., angle_indices])
        ds_state[...] = data_state.astype(ds_state.dtype)

    print(f"[√] 已将 {hdf5_path} 中的角度（列1-6、8-13）从度转换为弧度")

def add_action_fields(hdf5_path: str):
    with h5py.File(hdf5_path, 'a') as f:
        if '/observation' not in f:
            raise KeyError("Group '/observation' 不存在！")
        state = f['/observation/state']
        T_joint = state.shape[0]
        actions = f.create_dataset('/action/actions', (T_joint, state.shape[1]), dtype=np.float32)

        actions[:-1] = state[1:]
        actions[-1] = state[-1]

    print(f"[√] 已成功在 '{hdf5_path}' 中添加 actions 数据集")

if __name__ == '__main__':
    HDF5_DIR = 'data/hdf5_0613'
    for i in range(30):
        hdf5_path = f'{HDF5_DIR}/episode_{i}.hdf5'
        convert_angle_to_radian(hdf5_path)
        add_action_fields(hdf5_path)