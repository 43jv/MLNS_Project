# test_model.py
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset import TestbedDataset
import metrics
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
from tqdm import tqdm
from datetime import datetime

# Import original and improved models
from model import DeepTTG
from improvedmodel import ImprovedDeepTTG, create_dataset_subset


def test(model, test_loader, loss_function, device, show=True):
    """Evaluate model performance on a data loader"""
    model.eval()
    test_loss = 0
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, data in tqdm(
            enumerate(test_loader), disable=not show, total=len(test_loader)
        ):
            data = data.to(device)
            y = data.y

            # Handle both model types - original and improved
            if isinstance(model, ImprovedDeepTTG):
                y_hat, _, _ = model(data)
            else:
                y_hat = model(data)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        "loss": test_loss,
        "c_index": metrics.c_index(targets, outputs),
        "RMSE": metrics.RMSE(targets, outputs),
        "MAE": metrics.MAE(targets, outputs),
        "SD": metrics.SD(targets, outputs),
        "CORR": metrics.CORR(targets, outputs),
    }

    return evaluation


def main():
    parser = argparse.ArgumentParser(description="Test DeepTGIN models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["original", "improved"],
        default="improved",
        help="Model type to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--subset",
        type=float,
        default=0.8,
        help="Fraction of dataset to use (for quick testing)",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create data loaders with subset for quick testing
    datasets = {}
    data_loaders = {}

    for phase_name in ["train", "val", "test2016", "test2013"]:
        print(f"Loading {phase_name} dataset...")
        dataset = TestbedDataset(root="data", dataset=phase_name)

        # Create subset for quicker testing
        if args.subset < 1.0:
            subset = create_dataset_subset(dataset, fraction=args.subset)
            print(f"Created subset of {len(subset)} samples from {len(dataset)} total")
            datasets[phase_name] = subset
        else:
            datasets[phase_name] = dataset

        data_loaders[phase_name] = DataLoader(
            datasets[phase_name],
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=(phase_name == "train"),
        )

    # Initialize model based on selection
    if args.model == "original":
        print("Using original DeepTTG model")
        model = DeepTTG().to(device)
    else:
        print("Using improved ImprovedDeepTTG model")
        model = ImprovedDeepTTG().to(device)

    # Loss function and optimizer
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Gradient scaler for mixed precision
    scaler = GradScaler()

    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0

        print(f"Epoch {epoch}/{args.epochs}")
        for batch_idx, data in enumerate(tqdm(data_loaders["train"], desc="Training")):
            data = data.to(device)
            optimizer.zero_grad()

            with autocast():
                if args.model == "original":
                    output = model(data)
                    loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
                else:
                    output, _, _ = model(data)
                    loss = loss_fn(output, data.y.view(-1, 1).float().to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Print progress for larger batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}, Loss: {loss.item()/len(data):.6f}")

        # Report epoch loss
        epoch_loss /= len(data_loaders["train"].dataset)
        print(f"Epoch {epoch} training loss: {epoch_loss:.6f}")

        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_metrics = test(model, data_loaders["val"], loss_fn, device)
        print(f"Validation metrics: {val_metrics}")

    # Final evaluation on test sets
    print("\n=== FINAL EVALUATION ===")
    results = {}

    for phase_name in ["train", "val", "test2016", "test2013"]:
        print(f"\nEvaluating on {phase_name} set:")
        performance = test(model, data_loaders[phase_name], loss_fn, device)
        results[phase_name] = performance

        print(f"{phase_name} results:")
        for k, v in performance.items():
            print(f"{k}: {v:.6f}")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"result/evaluation_{args.model}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    with open(f"{result_dir}/results.txt", "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Subset fraction: {args.subset}\n")
        f.write(f"Training epochs: {args.epochs}\n\n")

        for phase_name, performance in results.items():
            f.write(f"{phase_name}:\n")
            for k, v in performance.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write("\n")

    print(f"\nResults saved to {result_dir}/results.txt")


if __name__ == "__main__":
    main()
