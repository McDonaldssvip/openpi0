import math
import time
import rclpy
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from PIL import Image
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDatasetMetadata

from api.ranger_api import WaypointNavigator
from api.realman_api import RealmanRobot, rm_thread_mode_e

np.set_printoptions(precision=4, suppress=True)

def make_train_example(repo_id, episode_index=0, idx = 0):
    root = HF_LEROBOT_HOME / repo_id
    meta = LeRobotDatasetMetadata(repo_id=repo_id,root=root,)

    episode_index = 0
    parquet_rel = meta.get_data_file_path(episode_index)
    parquet_path = root / parquet_rel

    hf_ds = load_dataset("parquet",data_files=str(parquet_path),split="train",)
    states = np.array((hf_ds[:]['state']))
    actions = np.array((hf_ds[:]['actions']))
    images = hf_ds[:]['image']
    zeros = hf_ds[0]['wrist_image']
    return states, actions, images, zeros


def radians_to_degrees(batch: np.ndarray) -> np.ndarray:
    """
    将 batch 中的第 0-5 列和 7-12 列从弧度转度，返回一个新数组。
    """
    angle_idxs = list(range(0, 6)) + list(range(7, 13))
    deg = batch.copy()
    deg[..., angle_idxs] = np.rad2deg(deg[..., angle_idxs])
    return deg


def degrees_to_radians(batch: np.ndarray) -> np.ndarray:
    """
    将 batch 中的第 0-5 列和 7-12 列从度转弧度，返回一个新数组。
    """
    angle_idxs = list(range(0, 6)) + list(range(7, 13))
    rad = batch.copy()
    rad[..., angle_idxs] = np.deg2rad(rad[..., angle_idxs])
    return rad


def plot_comparison(arr1: np.ndarray, arr2: np.ndarray, idx: int,n_cols: int = 6, out_dir: str = "./output"):
    """
    在每个维度上比较 arr1 与 arr2，画折线并保存
    arr1, arr2 : ndarray 形状均为 (samples, dimensions) 的数据；samples 一般是 10。
    idx : int 保存文件名里用来区分不同批次的索引。
    n_cols : int, optional 子图列数；行数会自动计算。
    out_dir : str, optional 图片保存目录。
    """
    assert arr1.shape == arr2.shape, "两个数组必须形状相同"
    num_dims = arr1.shape[1]

    n_rows = math.ceil(num_dims / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,figsize=(3 * n_cols, 3 * n_rows),squeeze=False)
    axes = axes.flatten()

    for i in range(num_dims):
        ax = axes[i]
        ax.plot(arr1[:, i], label='pred_actions', marker='o')
        ax.plot(arr2[:, i], label='real_actions', marker='x')
        ax.set_title(f'Dimension {i + 1}')
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Value')
        ax.legend()

    for j in range(num_dims, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    save_path = f"{out_dir}/plots_{idx}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved to {save_path}")

def get_realsense_frame(pipeline, idx):
    frames      = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("无法获取彩色帧")
    color_image = np.asanyarray(color_frame.get_data())

    img = Image.fromarray(color_image)  # mode='RGB'
    img.save(f"./output/realsense_frame_{idx}.png")
    image_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
    image_resized.save(f"./output/realsense_frame_resized_{idx}.png")
    image_array = np.array(image_resized)
    # img.show()
    wrist_image_array = np.zeros((256, 256, 3), dtype=np.uint8)
    return image_array, wrist_image_array


def train_inference():
    rclpy.init()
    left_arm = RealmanRobot('192.168.1.18', 8080, rm_thread_mode_e.RM_TRIPLE_MODE_E)
    right_arm = RealmanRobot('192.168.1.19', 8080)
    navigator = WaypointNavigator()

    left_arm.move_to_init_pose()
    left_arm.open_gripper()
    right_arm.move_to_init_pose()
    right_arm.open_gripper()

    client = websocket_client_policy.WebsocketClientPolicy(host="192.168.31.243", port=8000)

    states, actions, images, zeros = make_train_example(repo_id="zzh/ranger_0613", episode_index=0, idx=0)

    for i in range(0, 200, 10):
        input = {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(np.array(images[i]), 256, 256)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(np.array(zeros), 256, 256)),
            "observation/state": np.array(states[i]),
            "prompt": "move to the table and pick up the object"
        }
        action_chunk = client.infer(input)['actions']

        pred_actions = radians_to_degrees(action_chunk)
        real_actions = radians_to_degrees(actions[i:i + 10])
        plot_comparison(pred_actions, real_actions, i)

        # print(f"Predicted actions at index {i}: {pred_actions}")
        # print(f"Real actions at index {i}: {real_actions}")

        linear_x = np.mean(pred_actions[:, 14])
        linear_y = np.mean(pred_actions[:, 15])
        ang_z = np.mean(pred_actions[:, 16])
        linear_vel = {'x': linear_x, 'y': linear_y, 'z': 0.0}
        ang_vel =  {'x': 0.0, 'y': 0.0, 'z': ang_z}
        navigator.move_with_velocity(linear_vel, ang_vel, duration=1)

        EPS_JOINT   = 1e-1  
        EPS_GRIPPER = 1e-2

        for action in pred_actions:
            current_pose = np.concatenate([
                left_arm.get_joint_states(),
                [left_arm.get_gripper_state()],
                right_arm.get_joint_states(),
                [right_arm.get_gripper_state()]
            ])

            if np.allclose(action[:14], current_pose, atol=[EPS_JOINT]*6 + [EPS_GRIPPER] + [EPS_JOINT]*6 + [EPS_GRIPPER]):
                continue
            right_arm.execute_joint_trajectory([action[7:13]])
            right_arm.open_gripper() if action[13] > 0.5 else right_arm.close_gripper()
            left_arm.execute_joint_trajectory([action[:6]])
            left_arm.open_gripper()  if action[6]  > 0.5 else left_arm.close_gripper()

def realtime_inference():
    rclpy.init()
    pipeline = rs.pipeline()
    rs_config   = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    profile = pipeline.start(rs_config)
    sensor = profile.get_device().first_color_sensor()

    sensor.set_option(rs.option.enable_auto_exposure,       1)
    sensor.set_option(rs.option.enable_auto_white_balance,  1)

    for _ in range(60):
        pipeline.wait_for_frames()
        time.sleep(0.03)

    sensor.set_option(rs.option.enable_auto_exposure,       0)
    sensor.set_option(rs.option.enable_auto_white_balance,  0)
    
    left_arm = RealmanRobot('192.168.1.18', 8080, rm_thread_mode_e.RM_TRIPLE_MODE_E)
    right_arm = RealmanRobot('192.168.1.19', 8080)
    navigator = WaypointNavigator()

    left_arm.move_to_init_pose()
    left_arm.open_gripper()
    right_arm.move_to_init_pose()
    right_arm.open_gripper()

    client = websocket_client_policy.WebsocketClientPolicy(host="192.168.31.243", port=8000)

    for i in range(30):
        image, wrist_image= get_realsense_frame(pipeline, i)
        state = np.concatenate([
                left_arm.get_joint_states(),
                [left_arm.get_gripper_state()],
                right_arm.get_joint_states(),
                [right_arm.get_gripper_state()]
            ])
        state = degrees_to_radians(np.array(state))

        input = {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(np.array(image), 256, 256)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(np.array(wrist_image), 256, 256)),
            "observation/state": state,
            "prompt": "move to the table and pick up the object"
        }
        action_chunk = client.infer(input)['actions']
        pred_actions = radians_to_degrees(action_chunk)

        linear_x = np.mean(pred_actions[:, 14])
        linear_y = np.mean(pred_actions[:, 15])
        ang_z = np.mean(pred_actions[:, 16])
        linear_vel = {'x': linear_x, 'y': linear_y, 'z': 0.0}
        ang_vel =  {'x': 0.0, 'y': 0.0, 'z': ang_z}
        if linear_x >0.01 or linear_y > 0.01 or ang_z > 0.01:
            navigator.move_with_velocity(linear_vel, ang_vel, duration=1)

        EPS_JOINT   = 1e-1  
        EPS_GRIPPER = 1e-2

        for action in pred_actions:
            current_pose = np.concatenate([
                left_arm.get_joint_states(),
                [left_arm.get_gripper_state()],
                right_arm.get_joint_states(),
                [right_arm.get_gripper_state()]
            ])

            if np.allclose(action[:14], current_pose, atol=[EPS_JOINT]*6 + [EPS_GRIPPER] + [EPS_JOINT]*6 + [EPS_GRIPPER]):
                continue
            right_arm.execute_joint_trajectory([action[7:13]])
            right_arm.open_gripper() if action[13] > 0.5 else right_arm.close_gripper()
            left_arm.execute_joint_trajectory([action[:6]])
            left_arm.open_gripper()  if action[6]  > 0.5 else left_arm.close_gripper()
    pipeline.stop()
    
if __name__ == "__main__":
    realtime_inference()