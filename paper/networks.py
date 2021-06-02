import torch
import torch.nn.functional as F

class CIFARConvNet(torch.nn.Module):

    def __init__(self):
        super(CIFARConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, padding=2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, padding=2)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(32, 32, 5, padding=2)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.conv4 = torch.nn.Conv2d(32, 32, 5, padding=2)
        self.relu4 = torch.nn.ReLU(inplace=True)

        self.fc5 = torch.nn.Linear(2048, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = F.avg_pool2d(x, 2)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, bias=True, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, bias=True, padding=1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(12544, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
