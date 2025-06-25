import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(model, dataloader, criterion, device, edge_index):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for x_seq, y in dataloader:
            x_seq, y = x_seq.to(device), y.to(device)
            out = model(x_seq, edge_index)
            loss = criterion(out, y.float())
            total_loss += loss.item() * x_seq.size(0)

            preds = (out > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(out.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc


def train_one_epoch(model, dataloader, optimizer, criterion, device, edge_index):
    model.train()
    all_preds, all_probs, all_labels = [], [], []

    for x_seq, y in dataloader:
        x_seq, y = x_seq.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x_seq, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(out).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return loss.item(), acc, f1, auc


def train_stgcn(model, train_dataset, test_dataset, edge_index, **params):
    num_epochs = params.get("num_epochs")
    batch_size = params.get("batch_size")
    device = params.get("device")
    learning_rate = params.get("learning_rate")


    model = model.to(device)
    edge_index = edge_index.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters in STGCN model: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    all_labels = torch.tensor(all_labels)
    num_pos = (all_labels == 1).sum()
    num_neg = (all_labels == 0).sum()
    pos_weight = num_neg / num_pos
    logging.info(f"pos_weight: {pos_weight:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_auc": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
        "test_auc": []
    }

    logging.info("Model Architecture:")
    logging.info(model)
    logging.info(f"Optimizer: {optimizer}")
    logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    logging.info(f"Loss function: BCEWithLogitsLoss with pos_weight={pos_weight.item():.4f}")
    logging.info(f"Device: {device}")
    logging.info(f"Number of nodes per graph: {train_dataset[0][0].shape[1]}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, edge_index)
        test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device, edge_index)
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["train_auc"].append(train_auc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)
        history["test_auc"].append(test_auc)

        logging.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
                    f"Test Loss: {test_loss:.4f} Acc: {test_acc:.3f} F1: {test_f1:.3f} AUC: {test_auc:.3f}")
        
    return history
