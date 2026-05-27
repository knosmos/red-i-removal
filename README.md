# red i removal
Solves the longstanding problem of red i removal (pain of photographers everywhere circa 1995) by training a lightweight ML model to remove red lowercase i characters from images. The architecture is a basic U-Net model with two encoder convolutional blocks, max pool, deconvolution, skip connection, and two decoder convolutional blocks. We train on a synthetic dataset of images taken from ImageNet augmented by randomly placed red lowercase i
characters. Optimization is standard Adam + MSE loss.

![Model Visualization](outputs/visualization.png)