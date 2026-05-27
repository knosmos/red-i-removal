import matplotlib.pyplot as plt
import numpy as np
import torch

from synthetic_dataset import SyntheticDataset
from model import RedIRemover


def visualize_samples(inputs, targets, outputs):
    assert (
        len(inputs) == len(targets) == len(outputs)
    ), "Inputs, targets, and outputs must have the same length."
    num_samples = len(inputs)
    plt.figure(figsize=(9, 3 * num_samples))
    # plt.subplot(1, 3, 1)
    # plt.title("Input")
    # plt.subplot(1, 3, 2)
    # plt.title("Ground Truth")
    # plt.subplot(1, 3, 3)
    # plt.title("Output")
    for idx, (sample_input, sample_gt, sample_output) in enumerate(
        zip(inputs, targets, outputs)
    ):
        plt.subplot(num_samples, 3, idx * 3 + 1)
        plt.imshow(
            (sample_input.squeeze().permute(1, 2, 0) * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        if idx == 0:
            plt.title("Input")
        plt.subplot(num_samples, 3, idx * 3 + 2)
        plt.imshow(
            (sample_gt.squeeze().permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        )
        if idx == 0:
            plt.title("Ground Truth")
        plt.subplot(num_samples, 3, idx * 3 + 3)
        plt.imshow(
            (sample_output.squeeze().detach().permute(1, 2, 0) * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        if idx == 0:
            plt.title("Output")
    plt.tight_layout()
    plt.savefig("outputs/visualization.png")
    plt.show()


if __name__ == "__main__":
    dataset = SyntheticDataset("train_data_batch_2")
    model = RedIRemover()
    model.load_state_dict(torch.load("outputs/best_model.pth"))
    inputs, targets, outputs = [], [], []
    for i in range(10, 14):
        sample = dataset[i]
        sample_input = sample["input"].unsqueeze(0)
        sample_gt = sample["gt"].unsqueeze(0)
        with torch.no_grad():
            sample_output = model(sample_input)
        inputs.append(sample_input)
        targets.append(sample_gt)
        outputs.append(sample_output)
    visualize_samples(inputs, targets, outputs)
