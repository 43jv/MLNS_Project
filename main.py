import os
import torch
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

import metrics
from model import DeepTTG
from improvedmodel import ImprovedDeepTTG, create_dataset_subset


class ProcessedDataset(InMemoryDataset):
    """Load a pre-saved <split>.pt containing (data, slices)."""

    def __init__(self, root, split):
        super().__init__(root)
        path = os.path.join(root, f"{split}.pt")
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass


def test(model, loader, loss_fn, device, show=True):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    with torch.no_grad():
        for batch in tqdm(loader, disable=not show):
            batch = batch.to(device)
            y_true = batch.y.view(-1)
            if isinstance(model, ImprovedDeepTTG):
                y_pred, _, _ = model(batch)
                y_pred = y_pred.view(-1)
            else:
                y_pred = model(batch).view(-1)

            total_loss += loss_fn(y_pred, y_true).item()
            preds.append(y_pred.cpu().numpy())
            trues.append(y_true.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    avg_loss = total_loss / len(loader.dataset)

    return {
        "loss": avg_loss,
        "c_index": metrics.c_index(trues, preds),
        "RMSE": metrics.RMSE(trues, preds),
        "MAE": metrics.MAE(trues, preds),
        "SD": metrics.SD(trues, preds),
        "CORR": metrics.CORR(trues, preds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["original", "improved"], default="improved")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--subset", type=float, default=0.1, help="Fraction of data to use")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # ─── Load data ───
    loaders = {}
    for split in ["train", "val", "test2016", "test2013"]:
        print(f"Loading {split}.pt …")
        ds = ProcessedDataset("data", split)
        if args.subset < 1.0:
            ds = create_dataset_subset(ds, args.subset)
            print(f" • Subsample: {len(ds)} / {len(ProcessedDataset('data', split))}")
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=(split == "train"), pin_memory=True
        )
    print()

    # ─── Model, Optimizer, Loss, AMP scaler ───
    if args.model == "original":
        model = DeepTTG().to(device)
    else:
        model = ImprovedDeepTTG().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    scaler = GradScaler()

    # ─── Training ───
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch}/{args.epochs}")

        for batch in tqdm(loaders["train"], desc="Training"):
            batch = batch.to(device)
            optimizer.zero_grad()

            with autocast():
                if args.model == "original":
                    y_pred = model(batch).view(-1)
                else:
                    y_pred, _, _ = model(batch)
                    y_pred = y_pred.view(-1)
                loss = loss_fn(y_pred, batch.y.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f" → Train loss: {running_loss/len(loaders['train'].dataset):.6f}")
        val_metrics = test(model, loaders["val"], loss_fn, device)
        print(f" → Val metrics: {val_metrics}\n")

    # ─── Final Evaluation ───
    print("=== FINAL EVALUATION ===")
    for split in ["train", "val", "test2016", "test2013"]:
        m = test(model, loaders[split], loss_fn, device)
        print(f"{split}:", m)

    # ─── Save results ───
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"result/eval_{args.model}_{ts}"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.txt"), "w") as f:
        f.write(
            f"Model: {args.model}\nSubset: {args.subset}\nEpochs: {args.epochs}\n\n"
        )
        for split in ["train", "val", "test2016", "test2013"]:
            perf = test(model, loaders[split], loss_fn, device)
            f.write(f"{split}:\n")
            for k, v in perf.items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")

    print(f"\nResults written to {out}/results.txt")


if __name__ == "__main__":
    main()
