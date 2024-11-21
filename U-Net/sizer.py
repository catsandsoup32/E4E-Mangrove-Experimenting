from original import ResNet_UNet
from densenet_unet_1 import DenseNet_UNet_1
from densenet_unet_2 import DenseNet_UNet_2
from densenet_unet_3 import DenseNet_UNet_3

def getParams(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

getParams(ResNet_UNet())
getParams(DenseNet_UNet_1())
getParams(DenseNet_UNet_2()) 
getParams(DenseNet_UNet_3())

