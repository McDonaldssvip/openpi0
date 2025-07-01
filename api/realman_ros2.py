import time
import rclpy
import threading
import rclpy.publisher

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from recorder_ros2 import Recorder
from realman_api import RealmanRobot
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e
'''
    ros2中通过Realman API实现双臂发布状态topic并录制rosbag的功能
        1. 发布关节状态
        2. 发布末端执行器状态
        3. 控制双臂并录制rosbag
'''

class Dual_arm_joint(Node):
    def __init__(self, left_robot: RealmanRobot, right_robot: RealmanRobot):
        super().__init__('realman_dualarm')

        self.left_robot = left_robot
        self.right_robot = right_robot
        cbg = ReentrantCallbackGroup()

        self.pub_js_left = self.create_publisher(JointState, 'left_arm/state', 10)
        self.create_timer(0.1, lambda: self.publish_state(self.left_robot, self.pub_js_left),callback_group=cbg)
        self.pub_js_right = self.create_publisher(JointState, 'right_arm/state', 10)
        self.create_timer(0.1, lambda: self.publish_state(self.right_robot, self.pub_js_right),callback_group=cbg)
    
    def publish_state(self, robot: RealmanRobot, pub_js: rclpy.publisher.Publisher):
        now = self.get_clock().now().to_msg()
        try:
            joints = robot.get_joint_states()
            grip = float(robot.get_gripper_state())
        except Exception as e:
            self.get_logger().warn(f'读取状态失败: {e}')
            return

        js = JointState()
        js.header.stamp = now
        js.name = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','gripper']
        js.position = joints + [grip]
        pub_js.publish(js)


class Dual_arm_eef(Node):
    def __init__(self, left_robot: RealmanRobot, right_robot: RealmanRobot):
        super().__init__('realman_dualarm')

        self.left_robot = left_robot
        self.right_robot = right_robot
        cbg = ReentrantCallbackGroup()

        self.pub_eef_left = self.create_publisher(JointState, 'left_arm/eef_states', 20)
        self.create_timer(0.05,lambda: self.publish_state(self.left_robot, self.pub_eef_left),callback_group=cbg)
        self.pub_eef_right = self.create_publisher(JointState, 'right_arm/eef_states', 20)
        self.create_timer(0.05,lambda: self.publish_state(self.right_robot, self.pub_eef_right),callback_group=cbg)
    
    def publish_state(self, robot: RealmanRobot, pub_eef:rclpy.publisher.Publisher):
        now = self.get_clock().now().to_msg()
        try:
            eef = robot.get_eef_states(1)
            grip = float(robot.get_gripper_state())
        except Exception as e:
            self.get_logger().warn(f'读取状态失败: {e}')
            return

        es = JointState()
        es.header.stamp = now
        es.name = ['eef_x','eef_y','eef_z','eef_r','eef_p','eef_y','gripper']
        es.position = eef + [grip]
        pub_eef.publish(es)


def example(args):
    right_1 = [-0.32293277978897095, -0.31710565090179443, 0.05417231470346451, 1.606262445449829, 0.042361896485090256, -0.8360847234725952]
    right_2 = [-0.3906264901161194, -0.3855360150337219, 0.049870651215314865, 1.6149039268493652, 0.030060315504670143, -0.8363309502601624]
    right_3 = [-0.3906420171260834, -0.3855139911174774, 0.24990655481815338, 1.6129344701766968, 0.029020681977272034, -0.8360002040863037]
    left_1 = [-0.3032529950141907, 0.2875870168209076, 0.2995781898498535, 1.5751521587371826, 0.036705780774354935, -2.2373578548431396]
    left_2 = [-0.37210574746131897, 0.35219600796699524, 0.31153619289398193, 1.6781706809997559, 0.04872271791100502, -2.2254183292388916]
    
    left_robot = RealmanRobot('192.168.1.19', 8080, rm_thread_mode_e.RM_TRIPLE_MODE_E)
    right_robot = RealmanRobot('192.168.1.18', 8080)
    
    rclpy.init(args=args)
    node = Dual_arm_joint(left_robot, right_robot)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    topics = [
        "/camera/outside/color/image_raw",
        "/left_arm/state",
        "/right_arm/state"]
    bag_path = "./rosbag_transfer_0516"

    left_robot.open_gripper()
    right_robot.open_gripper()
    left_robot.move_to_init_pose()
    right_robot.move_to_init_pose()
    for i in range(50):
        
        with Recorder(topics, bag_path, f'episode_{i}') as recorder:
            right_robot.execute_eef_trajectory([right_1])
            right_robot.execute_eef_trajectory([right_2])
            right_robot.close_gripper()
            right_robot.execute_eef_trajectory([right_3])
            left_robot.execute_eef_trajectory([left_1])
            left_robot.execute_eef_trajectory([left_2])
            left_robot.close_gripper()
            time.sleep(0.2)
            right_robot.open_gripper()
            time.sleep(0.2)
            right_robot.execute_relative_pose(0.08, 0.06, 0)
            right_robot.move_to_init_pose()

            left_robot.execute_relative_pose(0, 0, -0.197)
            left_robot.open_gripper()
        left_robot.execute_relative_pose(0.08, -0.06, 0)
        left_robot.move_to_init_pose()

    executor.shutdown()
    spin_thread.join()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    example()
