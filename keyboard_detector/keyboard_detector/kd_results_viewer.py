import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import ast 

class KDResultsViewer(Node):
    def __init__(self):
        super().__init__('kd_results_viewer')

        self.image_sub = self.create_subscription(
            Image,
            'zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10
        )
        self.keys_2d_sub = self.create_subscription(
            String,
            'keys_detected2d',
            self.keys_2d_callback,
            10
        )
        self.keys_3d_sub = self.create_subscription(
            String,
            'keys_detected3d',
            self.keys_3d_callback,
            10
        )
        self.bridge = CvBridge()
        self.current_image = None
        self.keys_2d = {}
        self.keys_3d = {}

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.display_results()
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def keys_2d_callback(self, msg):
        try:
            # Parse the dictionary-like string into a Python dictionary
            self.keys_2d = ast.literal_eval(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse keys_detected2d: {e}")

    def keys_3d_callback(self, msg):
        try:
            # Parse the dictionary-like string into a Python dictionary
            self.keys_3d = ast.literal_eval(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse keys_detected3d: {e}")

    def display_results(self):
        if self.current_image is not None:
            # Create a copy of the image to overlay text and points
            display_image = self.current_image.copy()

            # Overlay 2D key detections as scatter points
            for key, coords in self.keys_2d.items():
                x, y = int(coords[0]), int(coords[1])
                cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green dot
                if key in self.keys_3d:
                    # Overlay 3D key detections as text next to the point
                    cv2.putText(display_image, f"{key}: {self.keys_3d[key]}", (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Display the image
            cv2.imshow("KD Results Viewer", display_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = KDResultsViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()