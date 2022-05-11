

import sys
import os
from PIL import Image

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class filedialogdemo(QWidget):
    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        self.resize(500,500)
        layout = QVBoxLayout()
        self.btn = QPushButton("加载图片")
        self.btn.clicked.connect(self.getfile)
        layout.addWidget(self.btn)


        self.btn1 = QPushButton("加载本地摄像头")
        self.btn1.clicked.connect(self.getfiles)
        layout.addWidget(self.btn1)
        layout.addWidget(self.le)


        self.setLayout(layout)
        self.setWindowTitle("口罩识别系统")

    def getfile(self):
        '''
        getOpenFileName():返回用户所选择文件的名称，并打开该文件
        第一个参数用于指定父组件
        第二个参数指定对话框标题
        第三个参数指定目录
        第四个参数是文件扩展名过滤器
        '''

        self.fname, _  = QFileDialog.getOpenFileName(self, 'Open file',r'C:\Users\Administrator\Desktop\yolov5-5.0\data\images',"Image files (*.jpg *.gif)")
        self.le.setPixmap(QPixmap(self.fname))
        print(self.fname)  # 这个参数存储了选择进来的文件路径
        str=(r'python C:\Users\Administrator\Desktop\yolov5-5.0\detect2.py --source ' + self.fname+ ' --exist-ok ')
        os.system(str)  # 运行图片识别文件

        # 这里注意 如果在后面有程序错误，这里会直接自动结束程序 且不会报错
        path = os.listdir(r'C:\Users\Administrator\Desktop\yolov5-5.0\runs\detect\exp')
        s = path[0]
        pathend = r'C:\Users\Administrator\Desktop\yolov5-5.0\runs\detect\exp'+ '\\'+ s
        I = Image.open(pathend)
        I.show()



    def getfiles(self):   # 加载摄像头
        str=(r'python C:\Users\Administrator\Desktop\yolov5-5.0\detect.py ')   # python命令 + B.py + 参数：IC.txt'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 不加这个可能会报错
        os.system(str)













if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = filedialogdemo()
    ex.show()
    sys.exit(app.exec_())