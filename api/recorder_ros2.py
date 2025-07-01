import os
import time
import shutil
import signal
import subprocess

class Recorder:
    """
    Recorder 类，用于通过 ros2 bag record 命令录制指定的 ROS2 topic
    默认格式下录制结果为bag_path下一个rosbag为一个文件夹包含bag_name.db3 文件和 bag_name.yaml 文件
    此处处理保存格式为bag_path下包含所有db3和yaml文件
    """
    def __init__(self, topics, bag_path, bag_name, overwrite=True, extra_args=None):
        """
        :topics: 要记录的 topic 列表，例如 ["/topic1", "/topic2"]
        :bag_path: rosbag 输出目录，例如 "./data"
        :bag_name: 生成的 rosbag 基础文件名（不带扩展），例如 "1"，实际文件会是 "1.db3" 和 "1.yaml"
        :param overwrite: 是否覆盖已有同名 rosbag（默认 True）
        :param extra_args: 其他 ros2 bag record 参数列表
        """
        self.topics = topics
        self.bag_path = bag_path
        self.bag_name = bag_name
        self.overwrite = overwrite
        self.extra_args = extra_args or []
        self.process = None

    def _check_topics_exist(self):
        """
        检查所有指定的 topics 是否存在
        """
        result = subprocess.run(
            ['ros2', 'topic', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        available_topics = result.stdout.decode().splitlines()
        for topic in self.topics:
            if topic not in available_topics:
                print(f"[Recorder] 警告: 话题 {topic} 不存在")
                return False
        return True

    def start(self):
        if not self._check_topics_exist():
            exit()

        os.makedirs(self.bag_path, exist_ok=True)
        self._dir_prefix = os.path.join(self.bag_path, self.bag_name)
        if os.path.exists(self._dir_prefix) and self.overwrite:
            shutil.rmtree(self._dir_prefix)

        cmd = [
            "ros2", "bag", "record",
            "-o", self._dir_prefix,
        ] + self.extra_args + self.topics

        self.process = subprocess.Popen(cmd)
        time.sleep(0.5)
        print(f"[Recorder] started, recording to '{self._dir_prefix}'")

    def stop(self):
        if not self.process:
            print("[Recorder] Recorder 未启动或已停止")
            return

        self.process.send_signal(signal.SIGINT)
        self.process.wait()
        print(f"[Recorder] raw bag dir '{self._dir_prefix}' recording stopped")

        self._flatten_bag_dir()
        self.process = None

    def _flatten_bag_dir(self):
        """
        将 <bag_path>/<bag_name> 目录下的 .db3 与 metadata.yaml
        移到 <bag_path> 下，并分别重命名为<bag_name>.db3, <bag_name>.yaml 最后删除该子目录
        """
        src_dir = self._dir_prefix
        if not os.path.isdir(src_dir):
            print(f"[Recorder] 目录不存在，无法平铺：{src_dir}")
            return

        for fname in os.listdir(src_dir):
            full_src = os.path.join(src_dir, fname)
            if fname.endswith('.db3'):
                dst = os.path.join(
                    self.bag_path,
                    f"{self.bag_name}.db3"
                )
            elif fname.endswith('metadata.yaml') or fname.endswith('metadata.yml'):
                dst = os.path.join(
                    self.bag_path,
                    f"{self.bag_name}.yaml"
                )
            else:
                continue

            if os.path.exists(dst) and self.overwrite:
                os.remove(dst)

            shutil.move(full_src, dst)
            print(f"[Recorder] moved '{full_src}' -> '{dst}'")

        shutil.rmtree(src_dir)
        print(f"[Recorder] removed raw bag dir '{src_dir}'")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_tye, exc_val, exc_tb):
        self.stop()


def record_test():
    topics = ["/camera/image_raw",
              "/left_arm/eef_states",
              "/left_arm/joint_states",
              "/right_arm/eef_states",
              "/right_arm/joint_states"]
    bag_path = "./rosbag"
    for i in range(3):
        bag_name = f"episode_{i}"
        with Recorder(topics=topics, bag_path=bag_path, bag_name=bag_name, overwrite=True) as recorder:
            time.sleep(3)


if __name__ == '__main__':
    record_test()
