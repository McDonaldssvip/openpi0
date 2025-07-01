import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class OneFrameGrabber(Node):
    def __init__(self):
        super().__init__('one_frame_grabber')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/outside/color/image_raw', self.callback, 10)
        self.got_frame = False

    def callback(self, msg):
        if not self.got_frame:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite("frame.png", frame)
            self.get_logger().info("Saved one frame to frame.png")
            self.got_frame = True
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = OneFrameGrabber()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
