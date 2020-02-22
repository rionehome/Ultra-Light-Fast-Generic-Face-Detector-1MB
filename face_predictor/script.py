import argparse

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rione_msgs.srv import RequestFaceDetection

import numpy as np
import os

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../face_predictor/models/")


def create_predictor():
    parser = argparse.ArgumentParser(description='detect_video')

    parser.add_argument('--net_type', default="RFB", type=str,
                        help='The self.network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='define self.network input size,default optional value 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.7, type=float,
                        help='score threshold')
    parser.add_argument('--candidate_size', default=1000, type=int,
                        help='nms candidate size')
    parser.add_argument('--path', default="imgs", type=str,
                        help='imgs dir')
    parser.add_argument('--test_device', default="cpu", type=str,
                        help='cuda:0 or cpu')
    parser.add_argument('--video_path', default="/home/linzai/Videos/video/16_1.MP4", type=str,
                        help='path of video')
    args = parser.parse_args()

    input_img_size = args.input_size
    # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
    define_img_size(input_img_size)

    label_path = MODEL_PATH + "voc-model-labels.txt"

    net_type = args.net_type

    class_names = [name.strip() for name in open(label_path).readlines()]
    test_device = args.test_device

    candidate_size = args.candidate_size
    net = None
    model_path = None
    predictor = None
    if net_type == 'slim':
        model_path = MODEL_PATH + "pretrained/version-slim-320.pth"
        # model_path = "models/pretrained/version-slim-640.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
    elif net_type == 'RFB':
        model_path = MODEL_PATH + "pretrained/version-RFB-320.pth"
        # model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)

    net.load(model_path)

    return predictor


class FaceDetection(Node):
    def __init__(self):
        super().__init__("FaceDetection")

        self.predictor = create_predictor()
        self.create_subscription(Image, "/camera/color/image_raw", self.callback_image_realsence, 1)

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
