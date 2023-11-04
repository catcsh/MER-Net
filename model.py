import torch
import torch.nn as nn
import torch.nn.functional as F
from ResDeformCE import dcResidualBlock
from EGhost import GhostModule
from attention import SpatialAttention, ChannelAttention


class MER_Net(nn.Module):
    def __init__(self):
        super(MER_Net, self).__init__()
        self.conv11x11 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),     
            nn.MaxPool2d(kernel_size=3, stride=2)
        )   

        self.conv3x3_Channelto128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128), 
        )

        self.ghostConv1 = GhostModule(128, 128)

        self.dcResBlock1 = dcResidualBlock(128, 128)

        self.skModule = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128//16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128//16, 128, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        self.conv3x3_to256_to1024 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512), 
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024), 
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 38),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv11x11(x)
        x_Channelto128 = self.conv3x3_Channelto128(x)
        x_brach1_1 = self.ghostConv1(x_Channelto128)
        x_brach2_1 = self.dcResBlock1(x_Channelto128)

        x_brach1_add_brach2 = self.sigmoid(self.skModule(x_brach1_1 + x_brach2_1))

        x_brach1_2 = self.ghostConv1(x_Channelto128)
        x_brach2_2 = self.dcResBlock1(x_Channelto128)

        x_brach1_2 = x_brach1_2 * x_brach1_add_brach2
        x_brach2_2 = x_brach2_2 * x_brach1_add_brach2

        x = self.conv1x1(x_brach1_2 + x_brach2_2) + x_Channelto128

        x = self.conv3x3_to256_to1024(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    