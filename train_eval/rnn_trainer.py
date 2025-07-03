import torch
import torch.nn as nn
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    tot_loss, preds, probs, labs = 0., [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train: 
            optimizer.zero_grad()
        out = model(x)

        # print("x.shape:", x.shape)        # Expect [B, T, D]
        # print("y.shape:", y.shape)        # Expect [B]
        # print("y dtype:", y.dtype)        # Should be float32
        # print("model output shape:", out.shape)  # Expect [B]
        # print("model output (logits):", out[:5].detach().cpu().numpy())  # Look at sample logits
        # print("labels:", y[:5].detach().cpu().numpy())  # Look at sample labels
        loss = criterion(out, y)
        # print("Loss:", loss.item())
        # print("*" * 50)

        if train:
            loss.backward(); optimizer.step()
        tot_loss += loss.item() * x.size(0)
        preds.extend(((out > 0).int().cpu()).tolist())
        probs.extend(torch.sigmoid(out).detach().cpu().tolist())
        labs.extend(y.cpu().tolist())
    avg = tot_loss / len(loader.dataset)
    acc = accuracy_score(labs, preds)
    f1  = f1_score(labs, preds)
    auc = roc_auc_score(labs, probs)

    return avg, acc, f1, auc


def train_rnn(model, train_ds, test_ds, pos_weight, **params):
    device = params["device"]
    model  = model.to(device)
    batch_size = params.get("batch_size")
    lr = params.get("learning_rate")
    epochs = params.get("num_epochs")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [], "test_loss": [],
        "train_acc":  [], "test_acc": [],
        "train_f1":   [], "test_f1": [],
        "train_auc":  [], "test_auc": []
    }

    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        te = run_epoch(model, test_dl,  criterion, optimizer, device, train=False)
        scheduler.step(te[0])
        history["train_loss"].append(tr[0])
        history["train_acc"].append(tr[1])
        history["train_f1"].append(tr[2])
        history["train_auc"].append(tr[3])
        history["test_loss"].append(te[0])
        history["test_acc"].append(te[1])
        history["test_f1"].append(te[2])
        history["test_auc"].append(te[3])

        logging.info(
            f"Epoch {ep:02d} | Train Loss {tr[0]:.4f} Acc {tr[1]:.3f} F1 {tr[2]:.3f} | "
            f"Test Loss {te[0]:.4f} Acc {te[1]:.3f} F1 {te[2]:.3f} AUC {te[3]:.3f}"
        )

    return history
