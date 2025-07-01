import time
from Robotic_Arm.rm_robot_interface import RoboticArm,rm_thread_mode_e
'''
pip install Robotic_Arm==1.0.1
1.用rm_get_current_tool_frame检查EEF的坐标系是否一致，否则会出现同样的关节角度得到的EEF姿态相差一些角度
'''

class RealmanRobot:
    def __init__(self, ip, port, thread_mode=None):
        if thread_mode is None:
            self.robot = RoboticArm()
        else:
            self.robot = RoboticArm(mode=thread_mode)
        self.robot.rm_create_robot_arm(ip, port)
        time.sleep(0.2)

    def get_joint_states(self):
        '''
        返回6个关节的角度
        '''
        code, angles = self.robot.rm_get_joint_degree()
        return angles
    
    def get_eef_states(self, use_euler: int=1):
        '''
        返回EEF姿态，0表示四元数，1表示欧拉角
        '''
        code, angles = self.robot.rm_get_joint_degree()
        eef = self.robot.rm_algo_forward_kinematics(angles, use_euler)
        return eef

    def move_to_init_pose(self):
        '''
        初始化位置
        '''
        init_pose = [-1.36, 51.16, -144.32, 3.4, -83.13, -95.44]
        self.robot.rm_movej(init_pose, v=20, r=50, connect=0, block=True)

        
    def execute_joint_trajectory(self, trajectory_points):
        '''
        根据给定的多个关节角度运动
        '''
        for pose in trajectory_points:
            self.robot.rm_movej(pose, follow=False)


    def execute_eef_trajectory(self, eef_trajectory_points):
        '''
        根据给定的多个末端姿态运动
        '''
        for pose in eef_trajectory_points:
            self.robot.rm_movej_p(pose, v=20, r=50, connect=0, block=True)
            time.sleep(0.1)

    def execute_relative_pose(self, x, y, z):
        pose = self.get_eef_states()
        pose[0] += x
        pose[1] += y
        pose[2] += z
        self.robot.rm_movej_p(pose, v=20, r=50, connect=0, block=True)
        time.sleep(0.1)

    def get_gripper_state(self):
        '''
        夹爪打开角度1-1000,大于980认为打开，返回1；小于980认为关闭，返回0
        实际关闭夹爪读取角度会出现-2，-3等小于1的值
        '''
        code, state = self.robot.rm_get_gripper_state()
        state =1 if state['actpos'] > 980 else 0
        return state

    def open_gripper(self):
        state = self.robot.rm_set_gripper_release(speed=600, block=True, timeout=3)
        if state != 0:
           raise Exception("open failed")
        time.sleep(0.2)

    def close_gripper(self):
        state = self.robot.rm_set_gripper_pick_on(speed=600, force=600, block=True, timeout=3)
        if state == 1:
           raise Exception("close failed")
        time.sleep(0.2)


def api_test():
    left_arm  = RealmanRobot("192.168.1.18", 8080, thread_mode=rm_thread_mode_e.RM_TRIPLE_MODE_E)
    right_arm = RealmanRobot("192.168.1.19", 8080)

    print("left arm current eef states:", left_arm.get_eef_states())
    print("right arm current eef states:", right_arm.get_eef_states())

    left_arm.move_to_init_pose()
    right_arm.move_to_init_pose()

if __name__ == "__main__":
    api_test()
