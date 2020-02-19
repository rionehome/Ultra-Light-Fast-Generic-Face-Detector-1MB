import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class FaceDetection(Node):
    def __init__(self):
        super().__init__("FaceDetection")

        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.callback_image_realsence,
            10
        )

    def callback_image_realsence(self, msg):
        print(msg)


def main():
    rclpy.init()
    node = FaceDetection()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
