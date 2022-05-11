""" Detect people wearing masks in videos
"""
import time
from pathlib import Path

import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from covidMaskDetector.MTCNN.mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from covidMaskDetector.train import MaskDetector


def tagVideo(videopath):
    outputPath='Output_video'
    """ detect if persons in video are wearing masks or not
    """
    modelpath = 'covidMaskDetector/models/face_mask.ckpt'
    model = MaskDetector()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=device)['state_dict'],
                          strict=False)
    
    model = model.to(device)
    model.eval()
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

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]

    # print(outputPath+'/'+str(videopath)+'_d.mp4')
    now = time.strftime("%Y%m%d_%H:%M:%S")
    # print(now)
    # print('{}'.format(now))
    writer = FFmpegWriter(filename=(outputPath+'/'+now+'.mp4'))

    for frame in vreader(str(videopath)):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        for i in range(bboxs.shape[0]):
            xbbox = bboxs[i, :4]
            xStart, yStart = int(xbbox[0]), int(xbbox[1])
            width, height = int((xbbox[2] - xbbox[0])), int((xbbox[3] - xbbox[1]))

            # predict mask label on extracted face
            faceImg = frame[yStart:yStart+height, xStart:xStart+width]
            output = model(transformations(faceImg).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)



            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)
            
            # center text according to the face frame
            textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2
            
            # draw prediction label
            cv2.putText(frame,
                        labels[predicted],
                        (textX, yStart-20),
                        font, 1, labelColor[predicted], 2)

        writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('main', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.close()
    cv2.destroyAllWindows()

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    tagVideo()
