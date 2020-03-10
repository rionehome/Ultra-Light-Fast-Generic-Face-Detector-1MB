import os

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from rione_msgs.msg import PredictResult
import yaml

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../face_predictor/models/")
YAML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../share/face_predictor/yaml/")


class FaceDetection(Node):
    def __init__(self):
        super().__init__("FaceDetection")
        self.param = yaml.load(open(os.path.join(YAML_DIR, "face_predictor.yaml")))["face_predictor"]["ros__parameters"]
        self.predictor = self.create_predictor(self.param)
        self.create_subscription(Image, self.param["ImageTopic"], self.callback_image, 1)
        self.pub_result = self.create_publisher(PredictResult, "/face_predictor/result", 10)

    @staticmethod
    def create_predictor(param):
        """
        推論器の作成
        :param param:
        :return:
        """
        net = None
        model_path = None
        predictor = None
        label_path = MODEL_PATH + "voc-model-labels.txt"
        # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
        define_img_size(param["InputSize"])
        class_names = [name.strip() for name in open(label_path).readlines()]
        if param["NetType"] == 'slim':
            model_path = MODEL_PATH + "pretrained/version-slim-320.pth"
            # model_path = "models/pretrained/version-slim-640.pth"
            net = create_mb_tiny_fd(len(class_names), is_test=True, device=param["TestDevice"])
            predictor = create_mb_tiny_fd_predictor(net, candidate_size=param["CandidateSize"],
                                                    device=param["TestDevice"])
        elif param["NetType"] == 'RFB':
            model_path = MODEL_PATH + "pretrained/version-RFB-320.pth"
            #model_path = MODEL_PATH + "pretrained/version-RFB-640.pth"
            net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=param["TestDevice"])
            predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=param["CandidateSize"],
                                                        device=param["TestDevice"])
        net.load(model_path)
        return predictor

    def callback_image(self, msg: Image):
        """
        画像のsubscribe
        :param msg:
        :return:
        """
        image_array = np.asarray(msg.data).reshape((480, 640, 3))
        boxes, labels, probs = self.predictor.predict(
            image_array,
            self.param["CandidateSize"] / 2,
            self.param["Threshold"]
        )
        points1 = []
        points2 = []
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            points1.append(Point(x=float(box[0]), y=float(box[1])))
            points2.append(Point(x=float(box[2]), y=float(box[3])))
            cv2.rectangle(image_array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
        cv2.imshow("color", image_array)
        cv2.waitKey(1)
        result = PredictResult()
        result.point1 = points1
        result.point2 = points2
        self.pub_result.publish(result)


def main():
    print("face detection")
    rclpy.init()
    node = FaceDetection()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
