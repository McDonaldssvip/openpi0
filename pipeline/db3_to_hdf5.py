import h5py
import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


def ros_image_to_numpy(msg):
    """
    Convert sensor_msgs/msg/Image to numpy.ndarray (H×W×C, uint8).
    """
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    channels = msg.step // msg.width
    img = arr.reshape(msg.height, msg.width, channels)
    # BGR->RGB conversion if needed (uncomment next line)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_rosbag(bag_path, topics_and_types):
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    tt = {t.name: t.type for t in reader.get_all_topics_and_types()}
    buffers = {topic: [] for topic in topics_and_types}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in topics_and_types:
            continue
        msg_cls = get_message(tt[topic])
        msg     = deserialize_message(data, msg_cls)

        if topics_and_types[topic] == 'sensor_msgs/msg/Image':
            img = ros_image_to_numpy(msg)
            buffers[topic].append((t, img))
        elif topics_and_types[topic] == 'sensor_msgs/msg/JointState':
            arr = np.array(msg.position, dtype=np.float32)
            buffers[topic].append((t, arr))
        elif topics_and_types[topic] == 'nav_msgs/msg/Odometry':
            # Extract x, y linear velocities and z angular velocity from Odometry
            lin_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
            ang_vel = np.array([msg.twist.twist.angular.z])
            odom_data = np.concatenate([lin_vel, ang_vel], axis=0)
            buffers[topic].append((t, odom_data))

    return buffers


def downsample_and_align(buffers, fps):
    for topic, buf in buffers.items():
        if len(buf) == 0:
            raise RuntimeError(f"Topic '{topic}' 没有读到任何消息，检查话题名称或 bag 文件录制情况！")

    ts_lists = {k: np.array([t for t, _ in v], dtype=np.int64) for k, v in buffers.items()}
    t0 = max(v[0] for v in ts_lists.values())
    t1 = min(v[-1] for v in ts_lists.values())
    dt = int(1e9 / fps)
    N = int((t1 - t0) // dt)
    timeline = t0 + np.arange(N, dtype=np.int64) * dt

    cam_topic = '/camera/outside/color/image_raw'
    joint_topics = ['/left_arm/state', '/right_arm/state']
    odom_topic = '/odom'

    aligned_imgs = []
    joint_state = []

    for t in timeline:
        tsi = ts_lists[cam_topic]
        idx = np.abs(tsi - t).argmin()
        aligned_imgs.append(buffers[cam_topic][idx][1])

        js = []
        for jt in joint_topics:
            ts_j = ts_lists[jt]
            i_j = np.abs(ts_j - t).argmin()
            js.append(buffers[jt][i_j][1])
        
        # Get the odometry data for this timestamp
        ts_o = ts_lists[odom_topic]
        i_o = np.abs(ts_o - t).argmin()
        odom = buffers[odom_topic][i_o][1]
        
        # Concatenate joint states and odometry data (7 + 7 + 3)
        state = np.concatenate([js[0], js[1], odom], axis=0)
        joint_state.append(state)

    return np.stack(aligned_imgs), np.stack(joint_state)


def write_hdf5(out_path, images, joint_state):
    with h5py.File(out_path, 'w') as f:
        grp = f.create_group('observation')
        grp.create_dataset('image',  data=images,        dtype='uint8',  compression='gzip')
        grp.create_dataset('state',  data=joint_state,  dtype='float32', compression='gzip')


def convert_hdf5(db3_path, hdf5_path):
    topics_and_types = {
        '/camera/outside/color/image_raw': 'sensor_msgs/msg/Image',
        '/left_arm/state': 'sensor_msgs/msg/JointState',
        '/right_arm/state': 'sensor_msgs/msg/JointState',
        '/odom': 'nav_msgs/msg/Odometry',
    }
    buffers = read_rosbag(db3_path, topics_and_types)
    imgs, joints = downsample_and_align(buffers, fps=10)
    write_hdf5(hdf5_path, imgs, joints)
    print(f'✅ 已生成 {hdf5_path}，共 {imgs.shape[0]} 帧，state 维度 {joints.shape[1]}')


if __name__ == '__main__':
    db3_dir = 'data/rosbag_0613'
    hdf5_dir = 'data/hdf5_0613'
    for i in range(30):
        db3_path = f'{db3_dir}/episode_{i}.db3'
        hdf5_path = f'{hdf5_dir}/episode_{i}.hdf5'

        convert_hdf5(db3_path, hdf5_path)
