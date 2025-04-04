import rclpy
from rclpy.node import Node

import rclpy.subscription
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
import message_filters

from cv_bridge import CvBridge

from keyboard_detector import mask_finder, key_finder, utils
from visualization_msgs.msg import Marker
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

MIN_AREA = 100

def plot_3d_points(output):
    """
    Plots 3D points with labels using matplotlib.

    Args:
        output (dict): A dictionary where keys are labels and values are [X, Y, Z] coordinates.
    """
    # Filter out invalid points (e.g., containing NaN values)
    valid_points = {key: value for key, value in output.items() if not any(np.isnan(value))}

    # Extract data for plotting
    keys = list(valid_points.keys())
    x = [coord[0] for coord in valid_points.values()]
    y = [coord[1] for coord in valid_points.values()]
    z = [coord[2] for coord in valid_points.values()]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z, c='b', marker='o')

    # Add labels to each point
    for i, key in enumerate(keys):
        ax.text(x[i], y[i], z[i], key, fontsize=8)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


class Detector(Node):
    def __init__(self):
        super().__init__("detector")
        self.get_logger().info("Detector node has been started")
        self.get_logger().info(os.getcwd())

        self.image_sub = message_filters.Subscriber(self, Image, 'zed/zed_node/rgb/image_rect_color')
        self.info_sub = message_filters.Subscriber(self, CameraInfo,'zed/zed_node/rgb/camera_info')
        self.depth_sub = message_filters.Subscriber(self, Image, 'zed/zed_node/depth/depth_registered')

        self.time_synchronizer = message_filters.TimeSynchronizer([self.image_sub, self.info_sub, self.depth_sub], 10)
        self.time_synchronizer.registerCallback(self.listener_callback)

        self.publisher3d_ = self.create_publisher(String, '/detection_node/keys_detected3d', 11)
        self.publisher2d_ = self.create_publisher(String, '/detection_node/keys_detected2d', 11)
        self.publisher_visualization_ = self.create_publisher(Marker, '/detection_node/key_visualization', 11)

        self.sentence = "H"#AL-062".split("")
        self.key_publisher = self.create_publisher(Pose, f"/keys_detector/H", 11)

        ####setup the model
        mf = mask_finder.MaskFinder()#key_det_model_path=os.path.join("models","key_detection_model (1).pth"))
        self.key_det_model = mf.key_det_model 
        self.keyboard_bbox_model = mf.keyboard_bbox_model
        self.processor = mf.processor 
        self.input_points = mf.input_points
        self.step = mf.step
        self.patch_size = mf.patch_size
        self.device = mf.device

        self.kf = key_finder.KeyFinder(min_keys=50, 
                                probability_threshold=0.8,
                                min_key_size= 200,
                                key_displacement_distance=5e-2,
                                input_missing_keys=False,
                                use_gauss=False,
                                max_cluster_size=None,
                                check_space_eccentricity= True,
                                min_eccentricity = 0.2,
                                cluster_epsilon=np.inf)
        
                
    def listener_callback(self, msg, msg2, msg3):
        self.get_logger().info('Received image with width: "%s"' % msg.width) 
        try:
            img = CvBridge().imgmsg_to_cv2(msg, 'bgr8')
            depth_map = CvBridge().imgmsg_to_cv2(msg3, '32FC1')

            bbox = mask_finder.keyboard_bbox(img, self.keyboard_bbox_model)  
            if bbox is None:
                self.get_logger().info('No keyboard')
                return

            self.get_logger().info('Keyboard detected: %s' % str(bbox))      

            patches = mask_finder.transform_image(img, bbox, self.patch_size, self.step)
            mask = mask_finder.mask_from_patches(patches, self.key_det_model, self.processor, self.input_points, self.patch_size, self.step, self.device)
            bbox_im = mask_finder.bbox_img(img, bbox)

            # # Display the mask and wait for user input
            # cv2.imshow("Mask", mask)
            # cv2.waitKey(0)  # Wait for a key press to proceed
            # cv2.destroyAllWindows()

            binary_mask = (mask > 0).astype(np.uint8) * 255  # Strict thresholding

            # cv2.imshow("Binary Mask", binary_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            min_area = MIN_AREA

            filtered_mask = np.zeros_like(binary_mask)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered_mask[labels == i] = 255
            mask = filtered_mask

            # cv2.imshow("Filtered Mask", filtered_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            combined_mask = mask_finder.combine_mask(bbox_im, mask)

            # # Display the mask and wait for user input
            # cv2.imshow("Filtered Mask", combined_mask)
            # cv2.waitKey(0)  # Wait for a key press to proceed
            # cv2.destroyAllWindows()

            img, overlay = utils.get_frame_overlay(bbox, combined_mask, img)

            keys = utils.retrieve_keys_from_mask(overlay)

            output = {}
            output2d = {}
            letters = self.kf.find(keys)

            self.get_logger().info('Keys detected: "%s"' % str(letters.keys().__len__()))
            for key, value in letters.items():
                y, x = self.kf.original_coords(key, value, img)
                x = int(x)
                y = int(y)
                K = np.array(msg2.k).reshape((3,3))
                fx = K[0,0]
                fy = K[1,1]
                cx = K[0,2]
                cy = K[1,2]
                Z = depth_map[x,y]
                X = ((x - cx) * Z) / fx
                Y = ((y - cy) * Z) / fy

                output[key] = [X,Y,Z]

                # if key == "H":
                #     pose = Pose()
                #     pose.position.x = float(X)
                #     pose.position.y = float(Y)
                #     pose.position.z = float(Z)
                #      self.key_publisher.publish(pose)

                output2d[key] = [y, x]
            marker = Marker()
            marker.header.frame_id = "zed_left_camera_optical_frame"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "key_detector"
            marker.id = 0
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.005  # diameter of spheres
            marker.scale.y = 0.005
            marker.scale.z = 0.005
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.get_logger().info('Keys detected: "%s"' % str(output))
            self.publisher3d_.publish(String(data=str(output)))
            self.publisher2d_.publish(String(data=str(output2d)))
            for key, value in output.items():
                x = value[0]
                y = value[1]
                z = value[2]
                if np.isnan(x):
                    continue
                if np.isnan(y):
                    continue
                if np.isnan(z):
                    continue
                point = Point()
                point.x = float(x)
                point.y = float(y)
                point.z = float(z)
                marker.points.append(point)
            self.publisher_visualization_.publish(marker)

            # try:
            #     plot_3d_points(output)
            # except Exception as e:
            #    self.get_logger().error('Error plotting 3D points: "%s"' % e)
        except Exception as e:
            self.get_logger().error('Error converting image: "%s"' % e)
            return

def main():
    rclpy.init()

    detector = Detector()

    rclpy.spin(detector)

    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
