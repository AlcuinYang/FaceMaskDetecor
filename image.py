""" Detect people wearing masks in videos
"""
import time

import cv2
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from covidMaskDetector.MTCNN.mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from covidMaskDetector.train import MaskDetector

def tagImg(imgpath):
    """ detect if persons in image are wearing masks or not
    """
    outputpath = 'Output_img'
    modelpath = 'covidMaskDetector/models/face_mask.ckpt'
    #original model
    p_model_path = "covidMaskDetector/MTCNN/original_model/pnet_epoch_train.pt"
    r_model_path = "covidMaskDetector/MTCNN/original_model/rnet_epoch_train.pt"
    o_model_path = "covidMaskDetector/MTCNN/original_model/onet_epoch_train.pt"


    model = MaskDetector()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=device)['state_dict'],
                          strict=False)

    model = model.to(device)
    model.eval()

    pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path,
                                        use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=[0.6, 0.7, 0.7])

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    # if outputPath:
    # writer = FFmpegWriter(str(outputPath))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]
    count = 0
    img = cv2.imread(imgpath)
    # cv2.imshow('img',img)

    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print(bboxs)
    # print(landmarks)
# bboxs is faces
#     pylab.imshow(img_bg)
    for i in range(bboxs.shape[0]):
        xbbox = bboxs[i, :4]
        xStart,yStart = int(xbbox[0]),int(xbbox[1])
        width,height = int((xbbox[2]-xbbox[0])),int((xbbox[3]-xbbox[1]))

        # exit()
        # predict mask label on extracted face
        faceImg = img[yStart:yStart + height, xStart:xStart + width]
        output = model(transformations(faceImg).unsqueeze(0).to(device))
        _, predicted = torch.max(output.data, 1)
        # if predicted != 0:
        #     typelabel = 1
        # print(xbbox)

        cv2.rectangle(img,
                      (xStart, yStart),
                      (xStart + width, yStart + height),
                      (126, 65, 64),
                      thickness=2)
        textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
        textX = xStart + width // 2 - textSize[0] // 2

        # draw prediction label
        cv2.putText(img,
                    labels[predicted],
                    (textX, yStart - 20),
                    font, 1, labelColor[predicted], 2)




    cv2.imshow('main', img)
    now = time.strftime("%Y%m%d_%H:%M:%S")

    cv2.imwrite(outputpath+'/'+now+'.jpg',img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    tagImg('/Users/yorki/Desktop/1.png')