import torch
import torch.nn as nn
import numpy as np
from torch.nn import Conv2d, Module, BatchNorm2d, ReLU, MaxPool2d
from torchvision.models import densenet121, DenseNet121_Weights
from torchgeo.models import resnet18, get_weight
from typing import Optional
from unet_helpers import Upsample, Decoder, Decoder_No_Upsample


class DenseNet_UNet_3(Module):
    """
    - https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.densenet121
    - DenseNet121 has around 8 million params compared to ResNet18's 11 million, and performs a few points better on ImageNet
    - The backbone implementation is very similar to that of ResNet (four layers)
    """

    def __init__(self, input_image_size=256, resnet_as_initial_conv: Optional[True] = None):
        super(DenseNet_UNet_3, self).__init__()
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT) 
        ResNet = resnet18(weights=get_weight("ResNet18_Weights.SENTINEL2_RGB_SECO"))
        
        if resnet_as_initial_conv:
            initial_conv = nn.Sequential(
                ResNet.conv1,
                ResNet.bn1, 
                nn.ReLU(),
                ResNet.maxpool,
            )
        else:
            initial_conv = nn.Sequential(
                densenet.features.conv0,
                densenet.features.norm0,
                densenet.features.relu0,
                densenet.features.pool0
            )

        '''
        Got rid of the final dense block (which has depth 1024 and 16 layers)
        Added ResNet first layer before the first dense block
        '''
        self.layer1 = nn.Sequential(
            initial_conv,
            ResNet.layer1,
        )

        self.layer2 = nn.Sequential(
            densenet.features.denseblock1,
            densenet.features.transition1 
        )

        self.layer3 = nn.Sequential(
            densenet.features.denseblock2,
            densenet.features.transition2
        )
        self.layer4 = nn.Sequential(
            densenet.features.denseblock3, 
            densenet.features.transition3
        )

        dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
        x = self.layer1(dummy_input) 
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)
        
        # Define feature dimensions
        feature_dim = x.shape[1] 
        half_dim = feature_dim // 2
        quarter_dim = feature_dim // 4
        eighth_dim = feature_dim // 8
        sixteenth_dim = feature_dim // 16
        
        # Center
        self.center = Decoder(feature_dim, int(feature_dim // 1.5), half_dim) 

        # Skip connections
        self.skip_conv1 = Conv2d(half_dim, half_dim, kernel_size=1) 
        self.skip_conv2 = Conv2d(quarter_dim, quarter_dim, kernel_size=1)
        self.skip_conv3 = Conv2d(eighth_dim, eighth_dim, kernel_size=1)

        #decoder
        self.decoder1 = Decoder(feature_dim, half_dim, quarter_dim)
        self.decoder2 = Decoder(half_dim, quarter_dim, eighth_dim)
        
        self.classification_head = nn.Sequential(
            Upsample(quarter_dim, eighth_dim, sixteenth_dim),
            Conv2d(sixteenth_dim, 1, kernel_size=2, padding=1),
            nn.Upsample(
              size=(input_image_size, input_image_size),
              mode="bilinear",
              align_corners=False,
            ),
            Conv2d(1, 1, kernel_size=3, padding=1) # smooth output
        )

    def forward(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image[:, :3, :, :]

        # Encode
        x1 = self.layer1(image) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3)      

        # Center
        x = self.center(x4) 

        # Decode
        x = torch.cat((x, self.skip_conv1(x3)), dim=1) 
        x = self.decoder1(x) 
        x = torch.cat((x, self.skip_conv2(x2)), dim=1) 
        x = self.decoder2(x) 
        x = torch.cat((x, self.skip_conv3(x1)), dim=1) 
        
        x = self.classification_head(x)                
        return x
    

if __name__ == "__main__":
    model = DenseNet_UNet_3()
    output = model(torch.randn(1, 3, 256, 256))
