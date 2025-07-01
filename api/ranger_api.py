import math
import time
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from transforms3d.euler import quat2euler

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__("waypoint_navigator")
        
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        self.pose = None
        self.yaw = 0.0
        self.rate = self.create_rate(10)  # ROS2 频率对象
        self.odom_received = False
        
        # 容差值
        self.pos_tol = 0.01
        self.yaw_tol = 0.005

        self.get_logger().info("导航节点已启动，等待 odom 数据...")

    def odom_callback(self, msg):
        """处理里程计回调"""
        try:
            self.pose = msg.pose.pose
            orientation = self.pose.orientation

            w = orientation.w
            x = orientation.x
            y = orientation.y
            z = orientation.z
            
            (_, _, self.yaw) = quat2euler([w, x, y, z], axes='sxyz')
            
            if not self.odom_received:
                self.get_logger().info("首次接收到里程计数据")
                self.odom_received = True
        except Exception as e:
            self.get_logger().error(f"处理里程计时出错: {str(e)}")

    def stop(self):
        """停止机器人"""
        cmd = Twist()
        self.pub.publish(cmd)
        self.get_logger().info("机器人已停止")


    def get_quaternion_pose(self, timeout=5.0):
        """
        获取机器人当前位置和姿态，如果没有收到数据会等待
        参数：timeout: 等待里程计数据的超时时间（秒）
        返回：
            position: [x, y, z] 列表，表示机器人在世界坐标系下的位置
            quaternion: [x, y, z, w] 列表，表示机器人在世界坐标系下的姿态四元数
            如果超时未收到里程计数据，返回 None, None
        """
        # 等待里程计数据
        start_time = time.time()
        while not self.odom_received and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().warn(f"等待里程计数据超时({timeout}秒)")
                return None, None
        
        if self.pose is None:
            self.get_logger().warn("里程计数据无效")
            return None, None
            
        position = [
            self.pose.position.x,
            self.pose.position.y,
            self.pose.position.z
        ]
        
        quaternion = [
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w
        ]
        
        return position, quaternion


    def get_euler_pose(self, timeout=5.0):
        """
        获取机器人当前位置和欧拉角姿态
        返回：
            position: [x, y, z] 列表，表示机器人在世界坐标系下的位置
            euler: [roll, pitch, yaw] 列表，表示机器人在世界坐标系下的欧拉角（弧度）
            如果超时未收到里程计数据，返回 None, None
        """
        start_time = time.time()
        while not self.odom_received and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().warn(f"等待里程计数据超时({timeout}秒)")
                return None, None
        
        if self.pose is None:
            self.get_logger().warn("里程计数据无效")
            return None, None
            
        position = [
            self.pose.position.x,
            self.pose.position.y,
            self.pose.position.z
        ]
        
        orientation = self.pose.orientation
        euler = quat2euler([orientation.w, orientation.x, orientation.y, orientation.z], axes='sxyz')
        print("位置和欧拉角姿态:", position, list(euler))
        return position, list(euler)

    def get_quaternion_pose(self, timeout=5.0):
        """
        获取机器人当前位置和姿态，如果没有收到数据会等待
        参数：
            timeout: 等待里程计数据的超时时间（秒）
        返回：
            position: [x, y, z] 列表，表示机器人在世界坐标系下的位置
            quaternion: [x, y, z, w] 列表，表示机器人在世界坐标系下的姿态四元数
            如果超时未收到里程计数据，返回 None, None
        """
        # 等待里程计数据
        start_time = time.time()
        while not self.odom_received and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                self.get_logger().warn(f"等待里程计数据超时({timeout}秒)")
                return None, None
        
        if self.pose is None:
            self.get_logger().warn("里程计数据无效")
            return None, None
            
        position = [
            self.pose.position.x,
            self.pose.position.y,
            self.pose.position.z
        ]
        
        quaternion = [
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w
        ]
        print("位置和四元数姿态:", position, quaternion)
        return position, quaternion

    def rotate_to_angle(self, target_yaw):
        """旋转到目标角度"""
        start_time = time.time()
        
        while rclpy.ok():
            # 计算角度差 (保持在 -π 到 π 之间)
            angle_diff = math.atan2(math.sin(target_yaw - self.yaw), math.cos(target_yaw - self.yaw))
    
            if time.time() - start_time > 10.0:  # 10秒超时
                self.get_logger().warn("旋转超时！")
                break

            if abs(angle_diff) < self.yaw_tol:
                self.get_logger().info("旋转完成")
                break

            cmd = Twist()
            cmd.angular.z = 1.5 * angle_diff
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.stop()
        time.sleep(0.3)  # 短暂暂停

    def move_straight_to(self, goal_x, goal_y):
        """移动到目标点"""
        start_time = time.time()
        
        while rclpy.ok():
            dx = goal_x - self.pose.position.x
            dy = goal_y - self.pose.position.y
            distance = math.hypot(dx, dy)

            if time.time() - start_time > 10.0:  # 10秒超时
                self.get_logger().warn("移动超时！")
                break

            if distance < self.pos_tol:
                self.get_logger().info("到达目标位置")
                break
            
            # 计算目标方向
            target_yaw = math.atan2(dy, dx)
            # 计算角度差 (保持在 -π 到 π 之间)
            yaw_diff = math.atan2(math.sin(target_yaw - self.yaw), 
                                 math.cos(target_yaw - self.yaw))

            cmd = Twist()
            
            cmd.linear.x = 0.4
            cmd.angular.z = 1.5 * yaw_diff  # 增大转向修正速度
            
            self.pub.publish(cmd)

            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 移动完成后停止
        self.stop()
        time.sleep(0.3)  # 短暂暂停

    def move_to_waypoint(self, x, y, final_yaw=None):
        """移动到航点"""
        if self.pose is not None:
            curr_x = self.pose.position.x
            curr_y = self.pose.position.y
            self.get_logger().info(f"当前位置: ({curr_x:.3f}, {curr_y:.3f})")
        
        self.get_logger().info(f"导航到目标位置: ({x:.3f}, {y:.3f})")
        
        # 1. 计算初始方向并旋转
        dx = x - self.pose.position.x
        dy = y - self.pose.position.y
        target_yaw = math.atan2(dy, dx)
        
        self.get_logger().info(f"第一步: 旋转朝向目标位置 (目标角度: {math.degrees(target_yaw):.1f}°)")
        self.rotate_to_angle(target_yaw)
        
        # 2. 直线移动到目标
        self.get_logger().info(f"第二步: 向目标移动")
        self.move_straight_to(x, y)
        
        # 3. 转到最终方向 (如果需要)
        if final_yaw is not None:
            self.get_logger().info(f"第三步: 转向最终朝向 (目标角度: {math.degrees(final_yaw):.1f}°)")
            self.rotate_to_angle(final_yaw)
        
        self.get_logger().info("已到达目标点")
        time.sleep(1.0)  # 暂停一下

    def move_to_euler_pose(self, waypoints):
        """使用欧拉角运行导航任务
        参数：
            waypoints: 航点列表，每个航点为 [x, y] 或 [x, y, yaw] 格式
        """
        # 等待接收到里程计数据
        self.get_logger().info("等待里程计数据...")
        while rclpy.ok() and not self.odom_received:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"开始导航 {len(waypoints)} 个航点")
        
        for i, wp in enumerate(waypoints):
            x, y = wp[:2]
            yaw = wp[2] if len(wp) == 3 else None
            
            self.get_logger().info(f"===== 导航到航点 {i+1}/{len(waypoints)} =====")
            self.move_to_waypoint(x, y, yaw)

        self.get_logger().info("====== 所有航点导航完成! ======")
        self.stop()

    def move_to_quaternion_pose(self, waypoints):
        """使用四元数运行导航任务
        参数：
            waypoints: 航点列表，每个航点为 [x, y] 或 [x, y, qx, qy, qz, qw] 格式
        """
        # 等待接收到里程计数据
        self.get_logger().info("等待里程计数据...")
        while rclpy.ok() and not self.odom_received:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"开始导航 {len(waypoints)} 个航点")
        
        for i, wp in enumerate(waypoints):
            x, y = wp[:2]
            
            # 如果提供了四元数姿态
            if len(wp) == 6:
                qx, qy, qz, qw = wp[2:]
                # 将四元数转换为欧拉角
                (_, _, yaw) = quat2euler([qw, qx, qy, qz], axes='sxyz')
            else:
                yaw = None
            
            self.get_logger().info(f"===== 导航到航点 {i+1}/{len(waypoints)} =====")
            self.move_to_waypoint(x, y, yaw)

        self.get_logger().info("====== 所有航点导航完成! ======")
        self.stop()

    def move_with_velocity(self, linear_vel, angular_vel, duration=None):
        """使用速度控制机器人运动
        
        参数：
            linear_vel: 线速度字典，包含xyz三个方向的速度 (单位: m/s)
                格式: {'x': float, 'y': float, 'z': float}
                例如: {'x': 0.2, 'y': 0, 'z': 0} 表示向前移动
            angular_vel: 角速度字典，包含xyz三个轴的角速度 (单位: rad/s)
                格式: {'x': float, 'y': float, 'z': float}
                例如: {'x': 0, 'y': 0, 'z': 0.5} 表示逆时针旋转
            duration: 可选参数，运动持续时间（秒）。
                     如果不指定，函数会一直发送速度指令直到调用stop()
        """
        cmd = Twist()
        
        # 设置线速度
        cmd.linear.x = linear_vel.get('x', 0.0)
        cmd.linear.y = linear_vel.get('y', 0.0)
        cmd.linear.z = linear_vel.get('z', 0.0)
        
        # 设置角速度
        cmd.angular.x = angular_vel.get('x', 0.0)
        cmd.angular.y = angular_vel.get('y', 0.0)
        cmd.angular.z = angular_vel.get('z', 0.0)

        start_time = time.time()
        
        while rclpy.ok():
            self.pub.publish(cmd)
            
            if duration is not None:
                if time.time() - start_time >= duration:
                    break

            rclpy.spin_once(self, timeout_sec=0.1)

        self.stop()
