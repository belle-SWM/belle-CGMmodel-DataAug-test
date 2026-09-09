"""

Pytorch model frame for the ECG Upside-Down prediction.

Project: Upside-down ECG signal classification
Create time: Feb. 22, 2023
Author: Benjamin


"""

import torch.nn as nn


class model_v1_1(nn.Module):  #
    def __init__(self):
        super(model_v1_1, self).__init__()
        self.conv_l1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=50, stride=1, padding=0),  # (?, 1024, 90)
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(num_features=128)
        )

        self.conv_l2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=50, stride=1, padding=0),  # (?, 1024, 90)
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=256)
        )

        self.conv_l3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=25, stride=2, padding=0),  # (?, 1024, 90)
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=512)
        )
        self.conv_l4 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=25, stride=2, padding=0),  # (?, 1024, 90)
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=1024)
        )
        self.FC_1 = nn.Sequential(
            nn.Linear(1024 * 17, 256),

            # nn.Linear(2500, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=256)
        )
        self.FC_2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=64)
        )
        self.FC_3 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=16)
        )
        self.FC_4 = nn.Sequential(
            nn.Linear(16, 1),
            nn.BatchNorm1d(num_features=1),
            nn.Sigmoid()  # 配 Binary Cross Entropy
        )

    def forward(self, x):
        x1 = self.conv_l1(x)  # x.shape = (?, 1024, 45)   ? = batch size
        x2 = self.conv_l2(x1)  # x.shape = (?, 256, 39)   ? = batch size
        x3 = self.conv_l3(x2)  # x.shape = (?, 256, 39)   ? = batch size
        x4 = self.conv_l4(x3)
        x_fc = x4.view(x4.size(0), -1)  # 重構張量維度: 把 batch 維度保留，其他維度合一
        x_fc = self.FC_1(x_fc)
        x_fc = self.FC_2(x_fc)
        x_fc = self.FC_3(x_fc)
        x_fc = self.FC_4(x_fc)
        return x_fc

