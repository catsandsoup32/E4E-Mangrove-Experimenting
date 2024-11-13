# E4E-Mangrove-Experimenting

# 11-13-2024
Worked on exchanging the ResNet backbone in the U-Net for a DenseNet backbone.
- DenseNet121 has four dense blocks, which easily fit into the existing four layers in the ResNet U-Net implementation
- The caveat is that I couldn't find DenseNet weights trained on any sort of geo data, so this model may take more epochs to converge compared to ResNet
- ResNet halves feature map size after every layer, but DenseNet layer 3 and 4 both are 8x8 (no transition after layer 4)
- In **U-Net/densenet_unet_1.py**, I manually added a transition layer, but this brings the bottleneck size to 4x4, which may be too small for our pixel segmentation task
- In **U-Net/densenet_unet_2.py**, the decode operation for center is not upsampled, which may reduce the effectiveness of the first skip connection

Further topics for classification: 
- Worth looking into better weight initialization (U-Net original paper section 3.0 emphasizes this)
- Could try learning rate scheduler during training
- Try a three block DenseNet 