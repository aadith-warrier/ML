import torch
from torchsummary import summary

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        torch.nn.BatchNorm2d(out_channels))
        self.downsample = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
                        torch.nn.BatchNorm2d(out_channels)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, layers):
        super(ResNet, self).__init__()
        self.blocks = []
        for i in layers:
            block = ResidualBlock(i[0], i[1])
            self.blocks.append(block)
        
    def forward(self, x):
        for i in self.blocks:
            x = i.forward(x)
        return x


layers = [(3, 64), (64, 128), (128, 256)]
model = ResNet(layers)

summary(model, (3,512,512))