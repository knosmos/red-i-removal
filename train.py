import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from synthetic_dataset import SyntheticDataset
from model import RedIRemover

import matplotlib.pyplot as plt


def train_model(data_fname, batch_size=32, num_epochs=10, learning_rate=1e-3):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = SyntheticDataset(data_fname)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = RedIRemover().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            inputs = batch["input"].to(device)
            targets = batch["gt"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["gt"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Sample visualization of one image
        sample = next(iter(val_loader))
        sample_input = sample["input"][0].unsqueeze(0).to(device)
        sample_gt = sample["gt"][0].unsqueeze(0)
        sample_output = model(sample_input)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.imshow(
            (sample_input.squeeze().permute(1, 2, 0) * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(
            (sample_gt.squeeze().permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        )
        plt.subplot(1, 3, 3)
        plt.title("Output")
        plt.imshow(
            (sample_output.squeeze().detach().permute(1, 2, 0) * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        plt.tight_layout()
        plt.savefig(f"outputs/epoch_{epoch+1}.png")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader)}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")


if __name__ == "__main__":
    train_model("train_data_batch_1", batch_size=32, num_epochs=10, learning_rate=1e-3)
