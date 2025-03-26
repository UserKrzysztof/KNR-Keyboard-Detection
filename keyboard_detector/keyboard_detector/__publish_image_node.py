import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from builtin_interfaces.msg import Time

import pyzed.sl as sl

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(Image, '/zed/zed_node/rgb/image_rect_color', 10)
        self.cam_info_publisher_ = self.create_publisher(CameraInfo, '/zed/zed_node/rgb/camera_info', 10)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.cap = cv2.VideoCapture('/home/krzysztof/machine_learning/keyboard_detection/KNR-Keyboard-Detection/node/input.mp4')
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        self.get_logger().info(f"{ret} {frame}")
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            
            
            cam_info_msg = CameraInfo()
            stamp =self.get_clock().now().to_msg()
            cam_info_msg.header.stamp = stamp
            cam_info_msg.header.frame_id = 'camera_frame'
            cam_info_msg.p = [1066.0, 0.0 ,1024.0, 0.0, 0.0, 1066.0, 540.0, 0.0, 0.0,0.0,1.0, 0.0]
            msg.header.stamp = stamp
            self.publisher_.publish(msg)
            self.cam_info_publisher_.publish(cam_info_msg)
            self.get_logger().info("Iamge posted")
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()