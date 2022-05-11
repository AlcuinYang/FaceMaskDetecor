""" Detect people wearing masks in videos
"""
from pathlib import Path

import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector


def tagVideo(videopath):
    outputPath = '/'
    """ detect if persons in video are wearing masks or not
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # import face detector
    p_model_path = "covidMaskDetector/MTCNN/original_model/pnet_epoch.pt"
    r_model_path = "covidMaskDetector/MTCNN/original_model/rnet_epoch.pt"
    o_model_path = "covidMaskDetector/MTCNN/original_model/onet_epoch.pt"

    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path,
                                        use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=[0.6, 0.7, 0.7])

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]

    for frame in vreader(str(videopath)):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxs, landmarks = mtcnn_detector.detect_face(frame)

        for i in range(bboxs.shape[0]):
            xbbox = bboxs[i, :4]
            xStart, yStart = int(xbbox[0]), int(xbbox[1])
            width, height = int((xbbox[2] - xbbox[0])), int((xbbox[3] - xbbox[1]))


            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)

        cv2.imshow('main', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    tagVideo()
