""" Detect people wearing masks in videos
"""
from pathlib import Path

import cv2
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from covidMaskDetector.MTCNN.mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from covidMaskDetector.train import MaskDetector


def realtime():
    """ detect if persons in real-time are wearing masks or not
    """
    # import face detector

    p_model_path = "covidMaskDetector/MTCNN/original_model/pnet_epoch_train.pt"
    r_model_path = "covidMaskDetector/MTCNN/original_model/rnet_epoch_train.pt"
    o_model_path = "covidMaskDetector/MTCNN/original_model/onet_epoch_train.pt"

    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path,
                                        use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=[0.6, 0.7, 0.7])

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        for i in range(bboxs.shape[0]):
            xbbox = bboxs[i, :4]
            xStart, yStart = int(xbbox[0]), int(xbbox[1])
            width, height = int((xbbox[2] - xbbox[0])), int((xbbox[3] - xbbox[1]))

            # predict mask label on extracted face
            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (64, 65, 126),
                          thickness=2)

            # center text according to the face frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('main', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    realtime()
