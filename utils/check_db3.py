import sqlite3
import numpy as np

from typing import List, Tuple
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import get_message_slot_types

np.set_printoptions(precision=9, suppress=True)

def inspect_rostopics(db3_path: str) -> List[Tuple[int, str, str, str]]:
    """
    列出 .db3 ROS 2 bag 文件中所有的话题（topics），并显示每个 topic 对应的 ROS 消息字段结构。
    Args:db3_path: ROS 2 bag 的 .db3 文件路径。
    Returns:一个列表，每个元素是一个 4 元组：(topic_id, topic_name, message_type, serialization_format)
    """
    conn = sqlite3.connect(db3_path)
    try:
        sql = """
            SELECT id, name, type, serialization_format
              FROM topics
            ORDER BY name ASC
        """
        rows = conn.execute(sql).fetchall()

        print(f"\nFound {len(rows)} topics in '{db3_path}':")
        for tid, name, typ, fmt in rows:
            print(f"  • [{tid:3d}] {name:<30s} | type = {typ:<40s} | format = {fmt}")
            try:
                msg_cls = get_message(typ)
                msg_inst = msg_cls()
                slots = get_message_slot_types(msg_inst)
                print("      Fields:")
                for fname, ftype in slots.items():
                    print(f"        - {fname}: {ftype}")
            except Exception as e:
                print(f"      [Warning] 无法获取消息类型结构: {e}")

        return rows
    finally:
        conn.close()


def show_rosbag_data(db3_path: str, topic_name: str, num_samples: int = 5, auto_decode: bool = True) -> None:
    conn = sqlite3.connect(db3_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, type, serialization_format FROM topics WHERE name = ?",
        (topic_name,)
    )
    row = cur.fetchone()
    if not row:
        print(f"⚠️ 没有找到 topic `{topic_name}`")
        conn.close()
        return
    tid, typ, fmt = row
    print(f"▶ Topic `{topic_name}` ({typ}, format={fmt})")

    cur.execute(
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id = ? ORDER BY timestamp ASC LIMIT ?",
        (tid, num_samples)
    )
    msgs = cur.fetchall()
    if not msgs:
        print("⚠️ 没有消息。")
        conn.close()
        return

    if auto_decode:
        MsgType = get_message(typ)

    for i, (ts, blob) in enumerate(msgs, 1):
        print(f"  [{i}] ts = {ts}, raw bytes = {len(blob)}")
        if auto_decode:
            try:
                msg = deserialize_message(blob, MsgType)

                # For Odometry message (nav_msgs/msg/Odometry)
                if 'Odometry' in typ:
                    pose = msg.pose.pose  # Pose part
                    twist = msg.twist.twist  # Twist part
                    
                    position = pose.position
                    orientation = pose.orientation
                    linear_velocity = twist.linear
                    angular_velocity = twist.angular
                    
                    print(f"       → position = x: {position.x}, y: {position.y}, z: {position.z}")
                    print(f"       → orientation = x: {orientation.x}, y: {orientation.y}, z: {orientation.z}, w: {orientation.w}")
                    print(f"       → linear velocity = x: {linear_velocity.x}, y: {linear_velocity.y}, z: {linear_velocity.z}")
                    print(f"       → angular velocity = x: {angular_velocity.x}, y: {angular_velocity.y}, z: {angular_velocity.z}")
            
            except Exception as e:
                print(f"       ❌ decode 失败：{e}")
    
    conn.close()


if __name__ == '__main__':
    db3_path = 'data/rosbag_test/test.db3'
    # inspect_rostopics(db3_path)
    show_rosbag_data(db3_path, '/odom', num_samples=600)