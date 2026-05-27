# red i removal
Solves the longstanding problem of red i removal (pain of photographers everywhere circa 1995) by training a lightweight ML model to removing red lowercase i characters from images. The architecture is a basic U-Net-like model with two encoder convolutional blocks, max pool, deconvolution, skip connection, and two decoder convolutional blocks. The model is trained on a synthetic dataset of images taken from ImageNet.

![Model Visualization](outputs/visualization.png)