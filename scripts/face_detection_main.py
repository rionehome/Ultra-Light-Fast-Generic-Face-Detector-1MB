import rclpy
from rclpy.node import Node


class FaceDetection(Node):
    def __init__(self):
        super().__init__("FaceDetection")


if __name__ == '__main__':
    rclpy.init()
    node = FaceDetection()
    rclpy.spin(node)
