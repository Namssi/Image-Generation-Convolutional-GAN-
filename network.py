import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()

	self.textfc1 = nn.Linear(1024, 512)
	self.textfc2 = nn.Linear(512, 256)
	self.textfc1_bn = nn.BatchNorm2d(512)
	self.textfc2_bn = nn.BatchNorm2d(256)

        self.conv1 = nn.DataParallel(nn.ConvTranspose2d(100+256, 512, kernel_size=4))
        self.conv2 = nn.DataParallel(nn.Conv2d(512, 256, kernel_size=3, padding=1))
        self.conv3 = nn.DataParallel(nn.Conv2d(256, 128, kernel_size=3, padding=1))
        self.conv4 = nn.DataParallel(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.conv5 = nn.DataParallel(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.conv6 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self, x, text, input_size):
        self.input_size = input_size

	text = F.leaky_relu(self.textfc1_bn(self.textfc1(text)), 0.2)
	text = F.leaky_relu(self.textfc2_bn(self.textfc2(text)), 0.2)

	x = torch.cat([x, text], dim=1)
	x = x.view(x.size(0), -1, 1,1)

        #x = x.view(-1, self.input_size)
        #x = x.view(self.input_size, -1, 1,1)
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.upsampling(x)

        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)

        return F.tanh(self.conv6(x))



class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size

	self.textfc1 = nn.Linear(1024, 512)
        self.textfc2 = nn.Linear(512, 256)
        self.textfc1_bn = nn.BatchNorm2d(512)
        self.textfc2_bn = nn.BatchNorm2d(256)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
	self.conv5 = nn.Conv2d(512+256, 512, kernel_size=3, stride=1, padding=1)
	self.conv6 = nn.Conv2d(512, 1, kernel_size=4, stride=1)#, padding=1)
	self.conv2_bn = nn.BatchNorm2d(128)
	self.conv3_bn = nn.BatchNorm2d(256)
	self.conv4_bn = nn.BatchNorm2d(512)

    def forward(self, x, text):
        #x = x.view(-1, self.input_size)

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)


	text = F.leaky_relu(self.textfc1_bn(self.textfc1(text)), 0.2)
        text = F.leaky_relu(self.textfc2_bn(self.textfc2(text)), 0.2)
        text = text.view(text.size(0), text.size(1), 1,1)
	text = text.repeat(1, 1, x.size(2), x.size(3))

	x = torch.cat([x, text], dim=1)
	x = F.leaky_relu(self.conv4_bn(self.conv5(x)), 0.2)	

        return F.sigmoid(self.conv6(x))








