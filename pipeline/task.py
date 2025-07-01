import time
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e

from api.recorder_ros2 import Recorder
from api.realman_api import RealmanRobot
from api.realman_ros2 import Dual_arm_joint
from api.ranger_api import WaypointNavigator

class navigation_grasp():
    def __init__(self):
        rclpy.init()
        self.left_arm = RealmanRobot('192.168.1.18', 8080, rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.right_arm = RealmanRobot('192.168.1.19', 8080)
        self.navigator = WaypointNavigator()

        self.arm_node = Dual_arm_joint(self.left_arm, self.right_arm)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.arm_node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()


    def get_current_pose(self):
        """
        获取当前机器人的位姿
        """
        try:
            left_pose = self.left_arm.get_eef_states()
            print(f"Left Arm Pose: {left_pose}")
            right_pose = self.right_arm.get_eef_states()
            print(f"Right Arm Pose: {right_pose}")
            ranger_pose = self.navigator.get_quaternion_pose()
            return left_pose, right_pose, ranger_pose
        except Exception as e:
            self.navigator.get_logger().error(f"获取当前位姿时出错: {str(e)}")
            return None, None, None


    def arm_to_init(self):
        """
        将双臂移动到初始位置
        """
        try:
            self.left_arm.open_gripper()
            time.sleep(0.2)
            self.right_arm.open_gripper()
            self.left_arm.move_to_init_pose()
            self.right_arm.move_to_init_pose()
        except Exception as e:
            self.navigator.get_logger().error(f"移动到初始位置时出错: {str(e)}")

    def execute_grasp(self, right_trajectory, left_trajectory):
        try:
            self.right_arm.execute_eef_trajectory(right_trajectory)
            self.left_arm.execute_eef_trajectory(left_trajectory)
        except Exception as e:
            self.navigator.get_logger().error(f"运行中出错: {str(e)}")


    def destroy(self):
        self.executor.shutdown()
        self.spin_thread.join()
        self.arm_node.destroy_node()
        self.navigator.destroy_node()
        rclpy.shutdown()
        print("节点销毁，ROS2已关闭")

def arm():
    robot = navigation_grasp()
    robot.left_arm.execute_eef_trajectory([
            [0.3339, -0.7810, 0.1443, 3.1370, -0.2379, -1.550]
        ])
    robot.left_arm.execute_relative_pose(0, -0.015, -0.09)
    # exit()
    robot.left_arm.close_gripper()
    robot.left_arm.execute_relative_pose(-0.12, 0.0, 0.0)
    robot.right_arm.execute_eef_trajectory([
        [0.2748, 0.7278, 0.1878, 3.1307, 0.5278, -1.5965]
    ])
    robot.right_arm.execute_relative_pose(0, 0.025, -0.045)
    time.sleep(0.1)
    robot.right_arm.close_gripper()

def release():
    robot = navigation_grasp()
    robot.right_arm.open_gripper()
    robot.right_arm.execute_relative_pose(0, -0.02, 0.045)
    robot.left_arm.execute_relative_pose(0.12, 0, 0)
    time.sleep(1)
    robot.left_arm.open_gripper()
    time.sleep(0.3)
    robot.left_arm.execute_relative_pose(0, 0.02, 0.08)
    robot.arm_to_init()

def main(i):
    robot = navigation_grasp()
    topics = [
        "/camera/outside/color/image_raw",
        "/odom",
        "/left_arm/state",
        "/right_arm/state",
    ]
    bag_path = "./rosbag_0613"
    robot.arm_to_init()
    linear_vel = {'x': 0.3, 'y': 0.0, 'z': 0.0}
    ang_vel =  {'x': 0.0, 'y': 0.0, 'z': 0.0}
    linear_vel2 = {'x': -0.3, 'y': 0.0, 'z': 0.0}
    ang_vel2 =  {'x': 0.0, 'y': 0.0, 'z': 0.0}

    with Recorder(topics, bag_path, f'episode_{i}') as recorder:

        robot.navigator.move_with_velocity(linear_vel, ang_vel, 4.0)
        time.sleep(1)

        robot.left_arm.execute_eef_trajectory([
            [0.3339, -0.770, 0.14, 3.1370, -0.2379, -1.550]
        ])
        time.sleep(0.1)
        robot.left_arm.execute_relative_pose(0, -0.02, -0.08)
        robot.left_arm.close_gripper()
        time.sleep(0.1)
        robot.left_arm.execute_relative_pose(-0.12, 0.0, 0.0)
        robot.right_arm.execute_eef_trajectory([
            [0.2748, 0.7278, 0.1878, 3.1307, 0.5278, -1.5965]
        ])
        robot.right_arm.execute_relative_pose(0, 0.02, -0.045)
        robot.right_arm.close_gripper()
    time.sleep(2)
    robot.right_arm.open_gripper()
    robot.right_arm.execute_relative_pose(0, -0.02, 0.045)
    robot.left_arm.execute_relative_pose(0.12, 0, 0)
    time.sleep(1)
    robot.left_arm.open_gripper()
    time.sleep(0.3)
    robot.left_arm.execute_relative_pose(0, 0.02, 0.08)
    robot.arm_to_init()
    robot.navigator.move_with_velocity(linear_vel2, ang_vel2, 4.0)
    robot.destroy()

def test():
    robot = navigation_grasp()
    robot.arm_to_init()
    robot.left_arm.execute_eef_trajectory([
            [0.3339, -0.770, 0.14, 3.1370, -0.2379, -1.550]
    ])
    robot.left_arm.execute_relative_pose(0, -0.02, -0.08)
    # robot.get_current_pose()
    robot.left_arm.close_gripper()
    robot.left_arm.execute_relative_pose(-0.12, 0.0, 0.0)
    robot.right_arm.execute_eef_trajectory([
        [0.2748, 0.7278, 0.1878, 3.1307, 0.5278, -1.5965]
    ])
    robot.right_arm.execute_relative_pose(0, 0.02, -0.045)

if __name__ == '__main__':
    # test()
    main(29)
    # release()
    # arm()