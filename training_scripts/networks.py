import torch.nn as nn
import torch.nn.functional as F
from IPython import embed



class largest(nn.Module):
    def __init__(self, nb_classes):
        super(largest, self).__init__()
        self.nb_classes = nb_classes
        self.conv1 = nn.Conv2d(3, 92, 5, padding=2)
        self.pool = nn.MaxPool2d(3, stride = 2)
        self.conv2 = nn.Conv2d(92, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 512, 5, padding=2)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.drop = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1024, nb_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

    def forward_feat_last_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        return x    # with 96x96 images this returns (1L, 256L, 17L, 17L) features

    def forward_all_conv_feat(self, x):
        f_conv1 = self.conv1(x)
        x = self.pool(F.relu(f_conv1))
        f_conv2 = self.conv2(x)
        x = self.pool(F.relu(f_conv2))
        f_conv3 = self.conv3(x)
        return (f_conv1, f_conv2, f_conv3)

    def forward_all_conv_feat_after_relu(self, x):
        f_conv1 = F.relu(self.conv1(x))
        x = self.pool(f_conv1)
        f_conv2 = F.relu(self.conv2(x))
        x = self.pool(f_conv2)
        f_conv3 = F.relu(self.conv3(x))
        return (f_conv1, f_conv2, f_conv3)
