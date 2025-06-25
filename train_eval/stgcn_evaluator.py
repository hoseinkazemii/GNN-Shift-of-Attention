import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import logging


def evaluate_stgcn(model, test_dataset, edge_index, **params):
    device = params.get("device")
    batch_size = params.get("batch_size")


    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_seq, y in test_loader:
            x_seq, y = x_seq.to(device), y.to(device)
            out = model(x_seq, edge_index.to(device))
            preds = (out > 0.5).int().cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    report = classification_report(all_labels, all_preds, target_names=["No Powerline", "Powerline"], digits=4)
    logging.info("\nClassification Report (on Test Set):\n" + report)

    return all_labels, all_preds
