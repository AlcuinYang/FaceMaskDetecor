import sys

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
#print the strucure

# 设计的一个网络结构
class YNet(nn.Module):
    def __init__(self):
        super(YNet, self).__init__()
        self.conv1 =convLayer1= nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),  # (6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 =convLayer2= nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),  # (16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.conv3 = convLayer3=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3,3)),
            nn.ReLU(),  # (16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.linearLayers =linearLayers= nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2)
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(120, 84),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Linear(84, 10)

        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(layer.weight)
    # 定义前向传播过程，输入为x
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)
        return out


# 实例化一个网络模型
model = YNet()
# 假设输入13张1*28*28的图片
dummy_input = torch.rand(13, 3, 100, 100)
with SummaryWriter(comment='YNet') as w:
    w.add_graph(model, (dummy_input, ))

