# red i removal
Solves the longstanding problem of red i removal (pain of photographers everywhere circa 1995) by training a lightweight ML model to remove red lowercase i characters from images. The architecture is a basic U-Net model with two encoder convolutional blocks, max pool, deconvolution, skip connection, and two decoder convolutional blocks. We train on a synthetic dataset of images taken from ImageNet augmented by randomly placed red lowercase i
characters. Optimization is standard Adam + MSE loss.

![Model Visualization](outputs/visualization.png)

## Usage
- `train.py` - trains the model on the synthetic dataset and saves the best model to `outputs/best_model.pth`. Expects a dataset .npz file `train_data_batch_1`.
- `test.py` - loads the best model and visualizes the results on a few samples from the dataset. Saves the visualization to `outputs/visualization.png`.
