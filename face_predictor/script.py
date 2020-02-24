import os

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
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

        print(yaml.load(open("{}/face_predictor.yaml".format(YAML_DIR))), flush=True)
        # self.create_subscription(Image, "/camera/color/image_raw", self.callback_image_realsence, 1)

    @staticmethod
    def create_predictor(param):
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
            # model_path = "models/pretrained/version-RFB-640.pth"
            net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=param["TestDevice"])
            predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=param["CandidateSize"],
                                                        device=param["TestDevice"])
        net.load(model_path)
        return predictor

    def callback_image_realsence(self, msg):
        image_array = np.asarray(msg.data).reshape((480, 640, 3))
        # cv2.imshow("window", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        boxes, labels, probs = self.predictor.predict(image_array, 1000 / 2, 0.85)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(image_array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
        cv2.imshow('annotated', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


def main():
    print("face detection")
    rclpy.init()
    node = FaceDetection()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
